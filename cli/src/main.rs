use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Instant;

const API_ENDPOINT: &str = "https://api.x.ai/v1/responses";
const MODEL: &str = "grok-4-1-fast-non-reasoning";
const REASONING_MODEL: &str = "grok-4-1-fast";

#[derive(Parser)]
#[command(name = "grok-ask")]
#[command(about = "CLI for xAI Grok API with web and X search", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Search query (shorthand for search command)
    #[arg(long)]
    search: Option<String>,

    /// Ask query (shorthand for ask command)
    #[arg(long)]
    ask: Option<String>,

    /// Think query (shorthand for ask-thinking command)
    #[arg(long)]
    think: Option<String>,

    /// Chat query without web search
    #[arg(long)]
    chat: Option<String>,

    /// X search query (shorthand for x-search command)
    #[arg(long)]
    x_search: Option<String>,

    /// X ask query (shorthand for x-ask command)
    #[arg(long)]
    x_ask: Option<String>,

    /// Previous response ID for follow-up
    #[arg(short = 'r', long)]
    response_id: Option<String>,

    /// Only include posts from these X handles (comma-separated, without @)
    #[arg(long, value_delimiter = ',')]
    allowed_handles: Option<Vec<String>>,

    /// Exclude posts from these X handles (comma-separated, without @)
    #[arg(long, value_delimiter = ',')]
    excluded_handles: Option<Vec<String>>,

    /// Start date for X search (YYYY-MM-DD)
    #[arg(long)]
    from_date: Option<String>,

    /// End date for X search (YYYY-MM-DD)
    #[arg(long)]
    to_date: Option<String>,

    /// Enable image understanding for X search
    #[arg(long)]
    enable_images: bool,

    /// Enable video understanding for X search
    #[arg(long)]
    enable_video: bool,

    /// Output format
    #[arg(short, long, default_value = "text")]
    output: OutputFormat,
}

#[derive(Subcommand)]
enum Commands {
    /// Quick web search with minimal thinking
    Search {
        query: String,
        #[arg(long, default_value = "10")]
        max_results: u32,
    },
    /// Get grounded answer with balanced reasoning
    Ask {
        query: String,
        #[arg(short = 'r', long)]
        response_id: Option<String>,
    },
    /// Deep reasoning for complex problems
    Think {
        query: String,
        #[arg(short = 'r', long)]
        response_id: Option<String>,
    },
    /// Chat without web search
    Chat {
        query: String,
        #[arg(short = 'r', long)]
        response_id: Option<String>,
    },
    /// Search X (Twitter) posts
    XSearch {
        query: String,
        #[arg(long, default_value = "10")]
        max_results: u32,
        /// Only include posts from these X handles (comma-separated, without @)
        #[arg(long, value_delimiter = ',')]
        allowed_handles: Option<Vec<String>>,
        /// Exclude posts from these X handles (comma-separated, without @)
        #[arg(long, value_delimiter = ',')]
        excluded_handles: Option<Vec<String>>,
        /// Start date (YYYY-MM-DD)
        #[arg(long)]
        from_date: Option<String>,
        /// End date (YYYY-MM-DD)
        #[arg(long)]
        to_date: Option<String>,
        /// Enable image understanding
        #[arg(long)]
        enable_images: bool,
        /// Enable video understanding
        #[arg(long)]
        enable_video: bool,
    },
    /// Get grounded answers from X (Twitter) posts
    XAsk {
        query: String,
        #[arg(short = 'r', long)]
        response_id: Option<String>,
        /// Only include posts from these X handles (comma-separated, without @)
        #[arg(long, value_delimiter = ',')]
        allowed_handles: Option<Vec<String>>,
        /// Exclude posts from these X handles (comma-separated, without @)
        #[arg(long, value_delimiter = ',')]
        excluded_handles: Option<Vec<String>>,
        /// Start date (YYYY-MM-DD)
        #[arg(long)]
        from_date: Option<String>,
        /// End date (YYYY-MM-DD)
        #[arg(long)]
        to_date: Option<String>,
        /// Enable image understanding
        #[arg(long)]
        enable_images: bool,
        /// Enable video understanding
        #[arg(long)]
        enable_video: bool,
    },
    // TODO: Add XThink command - deep reasoning with X search grounding (use_reasoning=true, use_x_search=true)
}

#[derive(Clone, Debug, clap::ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

// Request structures
#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Tool {
    WebSearch(WebSearchTool),
    XSearch(XSearchTool),
}

#[derive(Serialize)]
struct WebSearchTool {
    r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_image_understanding: Option<bool>,
}

#[derive(Serialize)]
struct XSearchTool {
    r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_x_handles: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    excluded_x_handles: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    from_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    to_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_image_understanding: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_video_understanding: Option<bool>,
}

#[derive(Serialize)]
struct GrokRequest {
    model: String,
    input: Vec<Message>,
    store: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<Tool>,
}

// Response structures
#[derive(Deserialize, Serialize, Debug)]
struct GrokResponse {
    id: Option<String>,
    status: Option<String>,
    output: Option<Vec<Output>>,
    usage: Option<Usage>,
    #[serde(default)]
    error: Option<ApiError>,
}

#[derive(Deserialize, Serialize, Debug)]
struct Output {
    r#type: String,
    content: Option<Vec<Content>>,
    results: Option<Vec<WebSearchResult>>,
}

#[derive(Deserialize, Serialize, Debug)]
struct Content {
    r#type: String,
    text: Option<String>,
    annotations: Option<Vec<Annotation>>,
}

#[derive(Deserialize, Serialize, Debug)]
struct Annotation {
    url: Option<String>,
    title: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
struct WebSearchResult {
    url: Option<String>,
    title: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
struct Usage {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
}

#[derive(Deserialize, Serialize, Debug)]
struct ApiError {
    message: Option<String>,
    code: Option<String>,
}

fn get_api_key() -> Result<String> {
    env::var("XAI_API_KEY").context(
        "XAI_API_KEY environment variable not set. Get your key from https://console.x.ai/",
    )
}

/// X search configuration
#[derive(Default)]
struct XSearchConfig {
    allowed_handles: Option<Vec<String>>,
    excluded_handles: Option<Vec<String>>,
    from_date: Option<String>,
    to_date: Option<String>,
    enable_images: bool,
    enable_video: bool,
}

async fn create_request(
    query: &str,
    previous_response_id: Option<&str>,
    system_instruction: Option<&str>,
    max_tokens: u32,
    use_web_search: bool,
    use_x_search: bool,
    x_search_config: Option<XSearchConfig>,
    use_reasoning: bool,
) -> Result<GrokResponse> {
    let api_key = get_api_key()?;
    let client = reqwest::Client::new();

    let mut messages = Vec::new();

    // Add system instruction if provided
    if let Some(instruction) = system_instruction {
        messages.push(Message {
            role: "system".to_string(),
            content: instruction.to_string(),
        });
    }

    // Add user query
    messages.push(Message {
        role: "user".to_string(),
        content: query.to_string(),
    });

    let mut tools = Vec::new();

    if use_web_search {
        tools.push(Tool::WebSearch(WebSearchTool {
            r#type: "web_search".to_string(),
            enable_image_understanding: None,
        }));
    }

    if use_x_search {
        let config = x_search_config.unwrap_or_default();
        tools.push(Tool::XSearch(XSearchTool {
            r#type: "x_search".to_string(),
            allowed_x_handles: config.allowed_handles,
            excluded_x_handles: config.excluded_handles,
            from_date: config.from_date,
            to_date: config.to_date,
            enable_image_understanding: if config.enable_images { Some(true) } else { None },
            enable_video_understanding: if config.enable_video { Some(true) } else { None },
        }));
    }

    let model = if use_reasoning { REASONING_MODEL } else { MODEL };

    let request = GrokRequest {
        model: model.to_string(),
        input: messages,
        store: true,
        max_output_tokens: Some(max_tokens),
        previous_response_id: previous_response_id.map(|s| s.to_string()),
        tools,
    };

    let start = Instant::now();
    let response = client
        .post(API_ENDPOINT)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .context("Failed to send request")?;

    let elapsed = start.elapsed();
    eprintln!("Request completed in {:.2}s", elapsed.as_secs_f64());

    let data: GrokResponse = response.json().await.context("Failed to parse response")?;
    Ok(data)
}

fn format_response(response: &GrokResponse, format: &OutputFormat) -> String {
    match format {
        OutputFormat::Json => serde_json::to_string_pretty(response).unwrap_or_default(),
        OutputFormat::Text => {
            let mut output = String::new();

            // Check for error
            if let Some(error) = &response.error {
                output.push_str(&format!(
                    "Error: {}\n",
                    error.message.as_deref().unwrap_or("Unknown error")
                ));
                return output;
            }

            // Extract text from outputs
            let mut sources: Vec<(String, String)> = Vec::new();

            if let Some(outputs) = &response.output {
                for out in outputs {
                    if out.r#type == "message" {
                        if let Some(contents) = &out.content {
                            for content in contents {
                                if content.r#type == "output_text" || content.r#type == "text" {
                                    if let Some(text) = &content.text {
                                        output.push_str(text);
                                    }
                                    // Extract annotations
                                    if let Some(annotations) = &content.annotations {
                                        for ann in annotations {
                                            if let Some(url) = &ann.url {
                                                let title = ann.title.clone().unwrap_or_else(|| "Source".to_string());
                                                if !sources.iter().any(|(_, u)| u == url) {
                                                    sources.push((title, url.clone()));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else if out.r#type == "web_search_result" {
                        if let Some(results) = &out.results {
                            for result in results {
                                if let Some(url) = &result.url {
                                    let title = result.title.clone().unwrap_or_else(|| "Web Result".to_string());
                                    if !sources.iter().any(|(_, u)| u == url) {
                                        sources.push((title, url.clone()));
                                    }
                                }
                            }
                        }
                    } else if out.r#type == "x_search_result" {
                        if let Some(results) = &out.results {
                            for result in results {
                                if let Some(url) = &result.url {
                                    let title = result.title.clone().unwrap_or_else(|| "X Post".to_string());
                                    if !sources.iter().any(|(_, u)| u == url) {
                                        sources.push((title, url.clone()));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Add sources
            if !sources.is_empty() {
                output.push_str("\n\nSources:\n");
                for (i, (title, url)) in sources.iter().enumerate() {
                    output.push_str(&format!("{}. [{}]({})\n", i + 1, title, url));
                }
            }

            // Add follow-up instructions
            output.push_str("\n---\n");
            if let Some(id) = &response.id {
                output.push_str(&format!("To follow up, use response_id: {}\n", id));
            }

            output
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let result = if let Some(query) = &cli.search {
        let system_instruction = "Search for the query and return results in this exact format:\n\n---\nTITLE: [page title]\nURL: [full url]\nSNIPPET: [2-3 sentence excerpt]\n---\n\nReturn up to 10 results. No additional commentary or analysis.";
        create_request(
            query,
            cli.response_id.as_deref(),
            Some(system_instruction),
            4096,
            true,  // web search
            false, // no x search
            None,
            false, // no reasoning
        )
        .await?
    } else if let Some(query) = &cli.ask {
        create_request(
            query,
            cli.response_id.as_deref(),
            Some("Be concise and factual. Cite sources when using web information."),
            8192,
            true,  // web search
            false, // no x search
            None,
            false, // no reasoning
        )
        .await?
    } else if let Some(query) = &cli.think {
        create_request(
            query,
            cli.response_id.as_deref(),
            Some("Think step by step. Be thorough and cite sources."),
            16384,
            true,  // web search
            false, // no x search
            None,
            true,  // use reasoning model
        )
        .await?
    } else if let Some(query) = &cli.chat {
        create_request(
            query,
            cli.response_id.as_deref(),
            None,
            8192,
            false, // no web search
            false, // no x search
            None,
            false, // no reasoning
        )
        .await?
    } else if let Some(query) = &cli.x_search {
        let system_instruction = "Search X for the query and return results in this exact format:\n\n---\nAUTHOR: @[handle]\nPOST: [post content]\nURL: [full x.com url]\n---\n\nReturn up to 10 results. No additional commentary or analysis.";
        let config = XSearchConfig {
            allowed_handles: cli.allowed_handles.clone(),
            excluded_handles: cli.excluded_handles.clone(),
            from_date: cli.from_date.clone(),
            to_date: cli.to_date.clone(),
            enable_images: cli.enable_images,
            enable_video: cli.enable_video,
        };
        create_request(
            query,
            cli.response_id.as_deref(),
            Some(system_instruction),
            4096,
            false, // no web search
            true,  // x search
            Some(config),
            false, // no reasoning
        )
        .await?
    } else if let Some(query) = &cli.x_ask {
        let config = XSearchConfig {
            allowed_handles: cli.allowed_handles.clone(),
            excluded_handles: cli.excluded_handles.clone(),
            from_date: cli.from_date.clone(),
            to_date: cli.to_date.clone(),
            enable_images: cli.enable_images,
            enable_video: cli.enable_video,
        };
        create_request(
            query,
            cli.response_id.as_deref(),
            Some("Be concise and factual. Cite X posts when referencing discussions or opinions."),
            8192,
            false, // no web search
            true,  // x search
            Some(config),
            false, // no reasoning
        )
        .await?
    } else if let Some(command) = &cli.command {
        match command {
            Commands::Search { query, max_results } => {
                let system_instruction = format!(
                    "Search for the query and return results in this exact format:\n\n---\nTITLE: [page title]\nURL: [full url]\nSNIPPET: [2-3 sentence excerpt]\n---\n\nReturn up to {} results. No additional commentary or analysis.",
                    max_results
                );
                create_request(
                    query,
                    None,
                    Some(&system_instruction),
                    4096,
                    true,  // web search
                    false, // no x search
                    None,
                    false, // no reasoning
                )
                .await?
            }
            Commands::Ask { query, response_id } => {
                create_request(
                    query,
                    response_id.as_deref(),
                    Some("Be concise and factual. Cite sources when using web information."),
                    8192,
                    true,  // web search
                    false, // no x search
                    None,
                    false, // no reasoning
                )
                .await?
            }
            Commands::Think { query, response_id } => {
                create_request(
                    query,
                    response_id.as_deref(),
                    Some("Think step by step. Be thorough and cite sources."),
                    16384,
                    true,  // web search
                    false, // no x search
                    None,
                    true,  // use reasoning model
                )
                .await?
            }
            Commands::Chat { query, response_id } => {
                create_request(
                    query,
                    response_id.as_deref(),
                    None,
                    8192,
                    false, // no web search
                    false, // no x search
                    None,
                    false, // no reasoning
                )
                .await?
            }
            Commands::XSearch {
                query,
                max_results,
                allowed_handles,
                excluded_handles,
                from_date,
                to_date,
                enable_images,
                enable_video,
            } => {
                let system_instruction = format!(
                    "Search X for the query and return results in this exact format:\n\n---\nAUTHOR: @[handle]\nPOST: [post content]\nURL: [full x.com url]\n---\n\nReturn up to {} results. No additional commentary or analysis.",
                    max_results
                );
                let config = XSearchConfig {
                    allowed_handles: allowed_handles.clone(),
                    excluded_handles: excluded_handles.clone(),
                    from_date: from_date.clone(),
                    to_date: to_date.clone(),
                    enable_images: *enable_images,
                    enable_video: *enable_video,
                };
                create_request(
                    query,
                    None,
                    Some(&system_instruction),
                    4096,
                    false, // no web search
                    true,  // x search
                    Some(config),
                    false, // no reasoning
                )
                .await?
            }
            Commands::XAsk {
                query,
                response_id,
                allowed_handles,
                excluded_handles,
                from_date,
                to_date,
                enable_images,
                enable_video,
            } => {
                let config = XSearchConfig {
                    allowed_handles: allowed_handles.clone(),
                    excluded_handles: excluded_handles.clone(),
                    from_date: from_date.clone(),
                    to_date: to_date.clone(),
                    enable_images: *enable_images,
                    enable_video: *enable_video,
                };
                create_request(
                    query,
                    response_id.as_deref(),
                    Some("Be concise and factual. Cite X posts when referencing discussions or opinions."),
                    8192,
                    false, // no web search
                    true,  // x search
                    Some(config),
                    false, // no reasoning
                )
                .await?
            }
        }
    } else {
        eprintln!("No command or query provided. Use --help for usage.");
        std::process::exit(1);
    };

    println!("{}", format_response(&result, &cli.output));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test response parsing
    mod parse_response {
        use super::*;

        fn make_response(text: &str, sources: Vec<(&str, &str)>) -> GrokResponse {
            let results: Vec<WebSearchResult> = sources
                .iter()
                .map(|(title, url)| WebSearchResult {
                    title: Some(title.to_string()),
                    url: Some(url.to_string()),
                })
                .collect();

            GrokResponse {
                id: Some("resp_123".to_string()),
                status: Some("completed".to_string()),
                output: Some(vec![
                    Output {
                        r#type: "web_search_result".to_string(),
                        content: None,
                        results: Some(results),
                    },
                    Output {
                        r#type: "message".to_string(),
                        content: Some(vec![Content {
                            r#type: "output_text".to_string(),
                            text: Some(text.to_string()),
                            annotations: None,
                        }]),
                        results: None,
                    },
                ]),
                usage: None,
                error: None,
            }
        }

        #[test]
        fn test_format_simple_response() {
            let response = GrokResponse {
                id: Some("resp_123".to_string()),
                status: Some("completed".to_string()),
                output: Some(vec![Output {
                    r#type: "message".to_string(),
                    content: Some(vec![Content {
                        r#type: "output_text".to_string(),
                        text: Some("Hello, world!".to_string()),
                        annotations: None,
                    }]),
                    results: None,
                }]),
                usage: None,
                error: None,
            };

            let output = format_response(&response, &OutputFormat::Text);
            assert!(output.contains("Hello, world!"));
            assert!(output.contains("response_id: resp_123"));
        }

        #[test]
        fn test_format_response_with_sources() {
            let response = make_response(
                "Found results.",
                vec![("News", "https://news.com"), ("Blog", "https://blog.com")],
            );

            let output = format_response(&response, &OutputFormat::Text);
            assert!(output.contains("Found results."));
            assert!(output.contains("Sources:"));
            assert!(output.contains("[News](https://news.com)"));
            assert!(output.contains("[Blog](https://blog.com)"));
        }

        #[test]
        fn test_format_error_response() {
            let response = GrokResponse {
                id: None,
                status: Some("failed".to_string()),
                output: None,
                usage: None,
                error: Some(ApiError {
                    message: Some("Rate limit exceeded".to_string()),
                    code: Some("429".to_string()),
                }),
            };

            let output = format_response(&response, &OutputFormat::Text);
            assert!(output.contains("Error: Rate limit exceeded"));
        }

        #[test]
        fn test_format_json_output() {
            let response = GrokResponse {
                id: Some("resp_json".to_string()),
                status: Some("completed".to_string()),
                output: Some(vec![]),
                usage: None,
                error: None,
            };

            let output = format_response(&response, &OutputFormat::Json);
            assert!(output.contains("\"id\": \"resp_json\""));
            assert!(output.contains("\"status\": \"completed\""));
        }

        #[test]
        fn test_x_search_result_parsing() {
            let response = GrokResponse {
                id: Some("resp_x".to_string()),
                status: Some("completed".to_string()),
                output: Some(vec![
                    Output {
                        r#type: "x_search_result".to_string(),
                        content: None,
                        results: Some(vec![WebSearchResult {
                            title: Some("@user".to_string()),
                            url: Some("https://x.com/user/status/123".to_string()),
                        }]),
                    },
                    Output {
                        r#type: "message".to_string(),
                        content: Some(vec![Content {
                            r#type: "output_text".to_string(),
                            text: Some("X post found.".to_string()),
                            annotations: None,
                        }]),
                        results: None,
                    },
                ]),
                usage: None,
                error: None,
            };

            let output = format_response(&response, &OutputFormat::Text);
            assert!(output.contains("X post found."));
            assert!(output.contains("[@user](https://x.com/user/status/123)"));
        }
    }

    // Test request serialization
    mod serialize_request {
        use super::*;

        #[test]
        fn test_web_search_tool_serialization() {
            let tool = Tool::WebSearch(WebSearchTool {
                r#type: "web_search".to_string(),
                enable_image_understanding: None,
            });

            let json = serde_json::to_string(&tool).unwrap();
            assert!(json.contains("\"type\":\"web_search\""));
            assert!(!json.contains("enable_image_understanding"));
        }

        #[test]
        fn test_x_search_tool_serialization() {
            let tool = Tool::XSearch(XSearchTool {
                r#type: "x_search".to_string(),
                allowed_x_handles: Some(vec!["user1".to_string(), "user2".to_string()]),
                excluded_x_handles: None,
                from_date: Some("2025-01-01".to_string()),
                to_date: Some("2025-01-15".to_string()),
                enable_image_understanding: Some(true),
                enable_video_understanding: Some(true),
            });

            let json = serde_json::to_string(&tool).unwrap();
            assert!(json.contains("\"type\":\"x_search\""));
            assert!(json.contains("\"allowed_x_handles\":[\"user1\",\"user2\"]"));
            assert!(json.contains("\"from_date\":\"2025-01-01\""));
            assert!(json.contains("\"to_date\":\"2025-01-15\""));
            assert!(json.contains("\"enable_image_understanding\":true"));
            assert!(json.contains("\"enable_video_understanding\":true"));
        }

        #[test]
        fn test_x_search_tool_minimal() {
            let tool = Tool::XSearch(XSearchTool {
                r#type: "x_search".to_string(),
                allowed_x_handles: None,
                excluded_x_handles: None,
                from_date: None,
                to_date: None,
                enable_image_understanding: None,
                enable_video_understanding: None,
            });

            let json = serde_json::to_string(&tool).unwrap();
            assert!(json.contains("\"type\":\"x_search\""));
            assert!(!json.contains("allowed_x_handles"));
            assert!(!json.contains("from_date"));
        }

        #[test]
        fn test_request_serialization() {
            let request = GrokRequest {
                model: "grok-4-1-fast".to_string(),
                input: vec![Message {
                    role: "user".to_string(),
                    content: "test query".to_string(),
                }],
                store: true,
                max_output_tokens: Some(8192),
                previous_response_id: Some("resp_prev".to_string()),
                tools: vec![Tool::WebSearch(WebSearchTool {
                    r#type: "web_search".to_string(),
                    enable_image_understanding: None,
                })],
            };

            let json = serde_json::to_string(&request).unwrap();
            assert!(json.contains("\"model\":\"grok-4-1-fast\""));
            assert!(json.contains("\"store\":true"));
            assert!(json.contains("\"max_output_tokens\":8192"));
            assert!(json.contains("\"previous_response_id\":\"resp_prev\""));
            assert!(json.contains("\"type\":\"web_search\""));
        }

        #[test]
        fn test_request_without_tools() {
            let request = GrokRequest {
                model: "grok-4-1-fast-non-reasoning".to_string(),
                input: vec![Message {
                    role: "user".to_string(),
                    content: "chat".to_string(),
                }],
                store: true,
                max_output_tokens: None,
                previous_response_id: None,
                tools: vec![],
            };

            let json = serde_json::to_string(&request).unwrap();
            assert!(!json.contains("\"tools\""));
            assert!(!json.contains("\"max_output_tokens\""));
            assert!(!json.contains("\"previous_response_id\""));
        }
    }

    // Test XSearchConfig
    mod x_search_config {
        use super::*;

        #[test]
        fn test_default_config() {
            let config = XSearchConfig::default();
            assert!(config.allowed_handles.is_none());
            assert!(config.excluded_handles.is_none());
            assert!(config.from_date.is_none());
            assert!(config.to_date.is_none());
            assert!(!config.enable_images);
            assert!(!config.enable_video);
        }

        #[test]
        fn test_config_with_handles() {
            let config = XSearchConfig {
                allowed_handles: Some(vec!["user1".to_string(), "user2".to_string()]),
                excluded_handles: None,
                from_date: None,
                to_date: None,
                enable_images: false,
                enable_video: false,
            };

            assert_eq!(
                config.allowed_handles,
                Some(vec!["user1".to_string(), "user2".to_string()])
            );
        }

        #[test]
        fn test_config_with_dates() {
            let config = XSearchConfig {
                allowed_handles: None,
                excluded_handles: None,
                from_date: Some("2025-01-01".to_string()),
                to_date: Some("2025-01-15".to_string()),
                enable_images: false,
                enable_video: false,
            };

            assert_eq!(config.from_date, Some("2025-01-01".to_string()));
            assert_eq!(config.to_date, Some("2025-01-15".to_string()));
        }

        #[test]
        fn test_config_with_media() {
            let config = XSearchConfig {
                allowed_handles: None,
                excluded_handles: None,
                from_date: None,
                to_date: None,
                enable_images: true,
                enable_video: true,
            };

            assert!(config.enable_images);
            assert!(config.enable_video);
        }
    }

    // Integration tests with mocked HTTP
    mod integration {
        use super::*;

        #[tokio::test]
        async fn test_api_request_format() {
            // This test verifies the request structure without making real API calls
            let messages = vec![
                Message {
                    role: "system".to_string(),
                    content: "Be concise.".to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: "test query".to_string(),
                },
            ];

            let request = GrokRequest {
                model: MODEL.to_string(),
                input: messages,
                store: true,
                max_output_tokens: Some(8192),
                previous_response_id: None,
                tools: vec![
                    Tool::WebSearch(WebSearchTool {
                        r#type: "web_search".to_string(),
                        enable_image_understanding: None,
                    }),
                    Tool::XSearch(XSearchTool {
                        r#type: "x_search".to_string(),
                        allowed_x_handles: Some(vec!["elonmusk".to_string()]),
                        excluded_x_handles: None,
                        from_date: Some("2025-01-01".to_string()),
                        to_date: None,
                        enable_image_understanding: None,
                        enable_video_understanding: None,
                    }),
                ],
            };

            let json = serde_json::to_string_pretty(&request).unwrap();

            // Verify structure
            assert!(json.contains("grok-4-1-fast-non-reasoning"));
            assert!(json.contains("Be concise."));
            assert!(json.contains("test query"));
            assert!(json.contains("web_search"));
            assert!(json.contains("x_search"));
            assert!(json.contains("elonmusk"));
            assert!(json.contains("2025-01-01"));
        }

        #[tokio::test]
        async fn test_response_parsing() {
            let json = r#"{
                "id": "resp_test",
                "status": "completed",
                "output": [
                    {
                        "type": "x_search_result",
                        "results": [
                            {"url": "https://x.com/u/1", "title": "@user"}
                        ]
                    },
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "Test response."}
                        ]
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 20}
            }"#;

            let response: GrokResponse = serde_json::from_str(json).unwrap();

            assert_eq!(response.id, Some("resp_test".to_string()));
            assert_eq!(response.status, Some("completed".to_string()));

            let output = format_response(&response, &OutputFormat::Text);
            assert!(output.contains("Test response."));
            assert!(output.contains("[@user](https://x.com/u/1)"));
        }
    }
}
