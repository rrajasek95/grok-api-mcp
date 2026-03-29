# Grok API Skill

Use the Grok API MCP server to search the web and get grounded answers from Grok (xAI). Prefer the Rust CLI in `cli/` for direct terminal use; use the MCP server when an MCP host specifically needs tools.

## When to Use

- **Web search**: Current events, recent news, facts that need verification
- **Grounded answers**: Questions needing authoritative sources
- **Deep reasoning**: Complex problems requiring step-by-step analysis
- **General chat**: Conversations without web search

## Available Tools

### search - Quick Web Search
```
search(query: str, max_results: int = 10)
```
Returns structured search results (title, URL, snippet).

### ask - Grounded Answer
```
ask(query: str, response_id: Optional[str] = None, max_tokens: int = 8192)
```
Returns a concise, factual answer with citations.

### think - Deep Reasoning
```
think(query: str, response_id: Optional[str] = None, max_tokens: int = 16384)
```
Returns thorough analysis with step-by-step reasoning.

### chat - General Conversation
```
chat(query: str, response_id: Optional[str] = None, max_tokens: int = 8192)
```
Chat without web search for creative tasks or general conversation.

### x_search - Search X Posts
```
x_search(
    query: str,
    max_results: int = 10,
    allowed_handles: Optional[list[str]] = None,
    excluded_handles: Optional[list[str]] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    enable_images: bool = False,
    enable_video: bool = False,
)
```
Returns structured X search results with authors, posts, and URLs.

### x_ask - Grounded Answer From X
```
x_ask(
    query: str,
    response_id: Optional[str] = None,
    max_tokens: int = 8192,
    allowed_handles: Optional[list[str]] = None,
    excluded_handles: Optional[list[str]] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    enable_images: bool = False,
    enable_video: bool = False,
)
```
Returns an answer grounded in X posts, with follow-up support via `response_id`.

## Examples

### Search for current information
```
search("latest xAI announcements 2025")
```

### Ask a factual question
```
ask("What is xAI and what is their mission?")
```

### Deep analysis
```
think("Compare the capabilities and architecture of Grok vs other AI models")
```

### Search X posts
```
x_search("Latest xAI announcements", allowed_handles=["xai"])
```

### Ask using X posts as sources
```
x_ask("What are people saying about Grok?", from_date="2026-03-01")
```

### Multi-turn conversation
```
# First question
ask("What is quantum computing?")
# Returns: ... response_id: resp_abc123

# Follow-up (Grok remembers context)
ask("How is it different from classical computing?", response_id="resp_abc123")
```

## Output Format

Responses include:
- Answer text with inline citations
- Sources list (numbered URLs)
- Metadata: response_id, status, token usage

## Environment

Requires `XAI_API_KEY` environment variable. Get your key from https://console.x.ai/
