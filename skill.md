# Grok API Skill

Use the Grok API MCP to search the web and get grounded answers from Grok (xAI).

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

### ask_thinking - Deep Reasoning
```
ask_thinking(query: str, response_id: Optional[str] = None, max_tokens: int = 16384)
```
Returns thorough analysis with step-by-step reasoning.

### chat - General Conversation
```
chat(query: str, response_id: Optional[str] = None, max_tokens: int = 8192)
```
Chat without web search for creative tasks or general conversation.

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
ask_thinking("Compare the capabilities and architecture of Grok vs other AI models")
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
