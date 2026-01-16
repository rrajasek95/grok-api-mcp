# Grok API MCP

Two independent tools for stateful Grok interactions with web search grounding:

1. **CLI Client (Rust)** - Standalone `grok-ask` binary that calls Grok API directly
2. **MCP Server (Python)** - FastMCP backend for tool-based integrations (Claude Desktop, etc.)

## Features

- **Web Search Grounding**: Automatic web search for current/factual information
- **Stateful Conversations**: Maintain context across multiple queries via `response_id`
- **Multiple Tools**: Search, ask, think, and chat modes

## Architecture

The CLI and MCP Server are **independent** - use whichever fits your workflow:

```
┌─────────────────────────────────────────────────────────┐
│  CLI Client (Rust) - grok-ask                           │
│  - Standalone binary, no dependencies                   │
│  - Direct Grok API calls via HTTP                       │
│  - Use as Claude Code skill or command line             │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Grok API                                               │
│  - Models: grok-4-1-fast-non-reasoning / grok-4-1-fast  │
│  - Automatic grounding (web_search)                     │
│  - Server-side conversation state                       │
└─────────────────────┬───────────────────────────────────┘
                      ▲
                      │
┌─────────────────────┴───────────────────────────────────┐
│  MCP Server (Python/FastMCP) - optional                 │
│  - For Claude Desktop, Cursor, etc.                     │
│  - Same API features via MCP tools                      │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### CLI Usage

```bash
# Set API key
export XAI_API_KEY=your_key_here

# Quick search
grok-ask --search "latest AI news"

# Get grounded answer
grok-ask --ask "What is xAI?"

# Deep reasoning
grok-ask --think "Compare Grok to GPT-4"

# Chat without web search
grok-ask --chat "Tell me a joke"

# Follow-up conversation
grok-ask --ask "What products does xAI offer?" -r <response_id>

# JSON output
grok-ask --ask "Query" -o json
```

### MCP Server Usage

```bash
cd server
uv sync
cp .env.example .env
# Edit .env with your XAI_API_KEY
uv run python server.py
```

Then connect via MCP client with tools: `search`, `ask`, `think`, `chat`

## Installation

### CLI (Rust)

```bash
# Install globally
cd cli
cargo install --path .

# Or build locally
cargo build --release
# Binary at: ./target/release/grok-ask
```

### MCP Server (Python)

```bash
cd server
uv sync
cp .env.example .env
# Edit .env with your XAI_API_KEY
```

## API Key

Get your Grok API key from: https://console.x.ai/

## Tools

| Tool | Model | Web Search | Max Tokens |
|------|-------|------------|------------|
| `search` | grok-4-1-fast-non-reasoning | Yes | 4096 |
| `ask` | grok-4-1-fast-non-reasoning | Yes | 8192 |
| `think` | grok-4-1-fast | Yes | 16384 |
| `chat` | grok-4-1-fast-non-reasoning | No | 8192 |

## Usage Examples

### Search for current information
```python
search("latest AI news 2025")
```

### Ask a factual question
```python
ask("What is xAI and what products do they offer?")
```

### Deep analysis
```python
think("Compare Grok's capabilities to other AI models")
```

### General chat (no web search)
```python
chat("Tell me a joke about programming")
```

### Multi-turn conversation
```python
# First question
response = ask("What is quantum computing?")
# Returns: ... response_id: resp_abc123

# Follow-up (Grok remembers context)
response = ask("How is it different from classical computing?", response_id="resp_abc123")
```

## Stateful Conversations

The API maintains server-side state via `response_id`:

- No need to resend conversation history
- Implicit caching (faster, cheaper)
- Pass `response_id` from previous response to continue conversation

## Claude Desktop Integration

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "grok": {
      "command": "uv",
      "args": ["run", "python", "/path/to/grok-api-mcp/server/server.py"],
      "env": {
        "XAI_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

## Project Structure

```
grok-api-mcp/
├── cli/
│   ├── src/
│   │   └── main.rs        # Rust CLI
│   └── Cargo.toml
├── server/
│   ├── server.py          # MCP server (Python/FastMCP)
│   ├── pyproject.toml
│   └── .env.example
├── skill.md               # Claude Code skill
└── README.md
```

## API Reference

The server uses the Grok API endpoint: `https://api.x.ai/v1/responses`

### Request Format

```json
{
  "model": "grok-4-1-fast-non-reasoning",
  "input": [
    {"role": "user", "content": "Your question"}
  ],
  "tools": [
    {"type": "web_search"}
  ]
}
```

For `think` command, uses `grok-4-1-fast` (reasoning-capable model).

### Authentication

Bearer token authentication via `XAI_API_KEY` environment variable.

## Credits

Based on [gemini-interactions-mcp](https://github.com/DigiBugCat/gemini-interactions-mcp) by [@DigiBugCat](https://github.com/DigiBugCat).

## License

MIT
