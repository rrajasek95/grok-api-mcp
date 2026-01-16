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
#[command(about = "CLI for xAI Grok API with web search", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Search query (shorthand for search command)
    #[arg(long, conflicts_with = "ask_query", conflicts_with = "think_query", conflicts_with = "chat_query")]
    search: Option<String>,

    /// Ask query (shorthand for ask command)
    #[arg(long, conflicts_with = "search", conflicts_with = "think_query", conflicts_with = "chat_query")]
    ask: Option<String>,

    /// Think query (shorthand for ask-thinking command)
    #[arg(long, conflicts_with = "search", conflicts_with = "ask_query", conflicts_with = "chat_query")]
    think: Option<String>,

    /// Chat query without web search
    #[arg(long, conflicts_with = "search", conflicts_with = "ask_query", conflicts_with = "think_query")]
    chat: Option<String>,

    /// Previous response ID for follow-up
    #[arg(short = 'r', long)]
    response_id: Option<String>,

    /// Output format
    #[arg(short, long, default_value = "text")]
    output: OutputFormat,
}

#[derive(Subcommand)]
enum Commands {
    /// Quick search with minimal thinking
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
struct Tool {
    r#type: String,
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

async fn create_request(
    query: &str,
    previous_response_id: Option<&str>,
    system_instruction: Option<&str>,
    max_tokens: u32,
    use_web_search: bool,
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

    let tools = if use_web_search {
        vec![Tool { r#type: "web_search".to_string() }]
    } else {
        vec![]
    };

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
            true,
            false, // no reasoning
        )
        .await?
    } else if let Some(query) = &cli.ask {
        create_request(
            query,
            cli.response_id.as_deref(),
            Some("Be concise and factual. Cite sources when using web information."),
            8192,
            true,
            false, // no reasoning
        )
        .await?
    } else if let Some(query) = &cli.think {
        create_request(
            query,
            cli.response_id.as_deref(),
            Some("Think step by step. Be thorough and cite sources."),
            16384,
            true,
            true, // use reasoning model
        )
        .await?
    } else if let Some(query) = &cli.chat {
        create_request(
            query,
            cli.response_id.as_deref(),
            None,
            8192,
            false,
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
                    true,
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
                    true,
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
                    true,
                    true, // use reasoning model
                )
                .await?
            }
            Commands::Chat { query, response_id } => {
                create_request(
                    query,
                    response_id.as_deref(),
                    None,
                    8192,
                    false,
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
