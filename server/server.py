"""
Grok API MCP Server

A FastMCP server using the Grok API (xAI) for AI conversations with web and X search grounding.

Features:
- Web search grounding via Grok's built-in web_search tool
- X (Twitter) search grounding via Grok's x_search tool
- Prompt-based thinking for thorough analysis
- Stateful conversations via response_id
"""

import os
from typing import Optional
from fastmcp import FastMCP
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("Grok Research")

# Get API key from environment
XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    raise ValueError(
        "XAI_API_KEY environment variable is required. "
        "Get your API key from https://console.x.ai/"
    )

# API configuration
API_ENDPOINT = "https://api.x.ai/v1/responses"
MODEL = "grok-4-1-fast-non-reasoning"
REASONING_MODEL = "grok-4-1-fast"


def _create_request(
    input_content: str | list,
    previous_response_id: Optional[str] = None,
    max_tokens: int = 8192,
    system_instruction: Optional[str] = None,
    use_web_search: bool = True,
    use_x_search: bool = False,
    use_reasoning: bool = False,
    # X search filter parameters
    allowed_x_handles: Optional[list[str]] = None,
    excluded_x_handles: Optional[list[str]] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    enable_image_understanding: bool = False,
    enable_video_understanding: bool = False,
) -> dict:
    """
    Create a request to the Grok API.

    Returns parsed response with text, sources, response_id, and usage.

    X search filter parameters (only apply when use_x_search=True):
        allowed_x_handles: Only include posts from these X handles (max 10)
        excluded_x_handles: Exclude posts from these X handles (max 10)
        from_date: Start date in ISO8601 format (YYYY-MM-DD)
        to_date: End date in ISO8601 format (YYYY-MM-DD)
        enable_image_understanding: Allow model to analyze images in posts
        enable_video_understanding: Allow model to analyze videos in posts
    """
    # Build input messages
    messages = []

    if system_instruction:
        messages.append({
            "role": "system",
            "content": system_instruction
        })

    if isinstance(input_content, str):
        messages.append({
            "role": "user",
            "content": input_content
        })
    else:
        messages.extend(input_content)

    model = REASONING_MODEL if use_reasoning else MODEL

    payload = {
        "model": model,
        "input": messages,
        "store": True,  # Enable caching
    }

    # Add max_tokens if supported
    if max_tokens:
        payload["max_output_tokens"] = max_tokens

    # Build tools list
    tools = []

    # Add web search tool
    if use_web_search:
        web_search_tool = {"type": "web_search"}
        if enable_image_understanding:
            web_search_tool["enable_image_understanding"] = True
        tools.append(web_search_tool)

    # Add X search tool with optional filters
    if use_x_search:
        x_search_tool = {"type": "x_search"}

        if allowed_x_handles:
            x_search_tool["allowed_x_handles"] = allowed_x_handles[:10]  # Max 10
        if excluded_x_handles:
            x_search_tool["excluded_x_handles"] = excluded_x_handles[:10]  # Max 10
        if from_date:
            x_search_tool["from_date"] = from_date
        if to_date:
            x_search_tool["to_date"] = to_date
        if enable_image_understanding:
            x_search_tool["enable_image_understanding"] = True
        if enable_video_understanding:
            x_search_tool["enable_video_understanding"] = True

        tools.append(x_search_tool)

    if tools:
        payload["tools"] = tools

    # Add previous response for conversation continuity
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(API_ENDPOINT, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        return _parse_response(data)

    except httpx.HTTPStatusError as e:
        return {
            "error": f"API error: {e.response.status_code} - {e.response.text}",
            "response_id": None,
            "status": "failed"
        }
    except Exception as e:
        return {
            "error": f"Request failed: {str(e)}",
            "response_id": None,
            "status": "failed"
        }


def _parse_response(data: dict) -> dict:
    """Parse the Grok API response into a structured format."""
    result = {
        "response_id": data.get("id"),
        "status": data.get("status", "completed"),
        "text": "",
        "sources": [],
        "usage": data.get("usage", {})
    }

    # Handle output array
    for output in data.get("output", []):
        output_type = output.get("type")

        if output_type == "message":
            # Extract text content from message
            for content in output.get("content", []):
                if content.get("type") == "output_text":
                    result["text"] += content.get("text", "")
                elif content.get("type") == "text":
                    result["text"] += content.get("text", "")
                    # Extract annotations/citations if present
                    for ann in content.get("annotations", []):
                        source = {
                            "url": ann.get("url"),
                            "title": ann.get("title", "Source")
                        }
                        if source["url"] and source not in result["sources"]:
                            result["sources"].append(source)

        elif output_type == "web_search_call":
            # Capture web search query info
            pass

        elif output_type == "web_search_result":
            # Extract web search results
            for item in output.get("results", []):
                source = {
                    "url": item.get("url"),
                    "title": item.get("title", "Web Result")
                }
                if source["url"] and source not in result["sources"]:
                    result["sources"].append(source)

        elif output_type == "x_search_call":
            # Capture X search query info
            pass

        elif output_type == "x_search_result":
            # Extract X search results (posts)
            for item in output.get("results", []):
                source = {
                    "url": item.get("url"),
                    "title": item.get("title", item.get("author", "X Post"))
                }
                if source["url"] and source not in result["sources"]:
                    result["sources"].append(source)

    # Fallback: check for direct output text
    if not result["text"] and "output_text" in data:
        result["text"] = data.get("output_text", "")

    return result


def _format_response(result: dict) -> str:
    """Format the parsed result into a readable string."""
    if "error" in result:
        return f"Error: {result['error']}"

    output = [result.get("text", "")]

    # Add sources
    sources = result.get("sources", [])
    if sources:
        output.append("\n\nSources:")
        for i, source in enumerate(sources, 1):
            if isinstance(source, dict):
                title = source.get("title", "Untitled")
                url = source.get("url", "")
                output.append(f"{i}. [{title}]({url})")
            else:
                output.append(f"{i}. {source}")

    # Add follow-up instructions
    response_id = result.get("response_id")
    if response_id:
        output.append("\n---")
        output.append(f"To follow up, use response_id: {response_id}")

    return "\n".join(output)


# MCP Tools

@mcp.tool
def search(
    query: str,
    max_results: int = 10,
) -> str:
    """
    Quick web search using Grok. Returns structured search results.

    Best for finding web context around X/Twitter topics, verifying claims
    from X posts, or gathering sources to supplement X search results.

    Args:
        query: Search query
        max_results: Maximum number of results to return (default: 10)

    Returns:
        Structured search results with titles, URLs, and snippets
    """
    system_instruction = f"""Search for the query and return results in this exact format:

---
TITLE: [page title]
URL: [full url]
SNIPPET: [2-3 sentence excerpt]
---

Return up to {max_results} results. No additional commentary or analysis."""

    result = _create_request(
        input_content=query,
        system_instruction=system_instruction,
        max_tokens=4096,
        use_web_search=True,
    )

    return _format_response(result)


@mcp.tool
def ask(
    query: str,
    response_id: Optional[str] = None,
    max_tokens: int = 8192,
) -> str:
    """
    Get grounded answers from Grok with web search.

    Best for questions related to X/Twitter discourse, xAI products, or when
    you need web-grounded answers that complement X search results.
    To follow up on a previous response, pass the response_id from that response.

    Args:
        query: Your question
        response_id: Pass the response_id from a previous response to continue that conversation
        max_tokens: Maximum response length (default: 8192)

    Returns:
        Answer with sources. Use the returned response_id to ask follow-up questions.
    """
    result = _create_request(
        input_content=query,
        previous_response_id=response_id,
        max_tokens=max_tokens,
        system_instruction="Be concise and factual. Cite sources when using web information.",
        use_web_search=True,
    )

    return _format_response(result)


@mcp.tool
def think(
    query: str,
    response_id: Optional[str] = None,
    max_tokens: int = 16384,
) -> str:
    """
    Deep reasoning with Grok for complex problems related to X/Twitter ecosystem.

    Uses step-by-step reasoning with web grounding. Best for analyzing X/Twitter
    trends, understanding social media dynamics, or complex questions where
    X discourse is relevant context.
    To follow up on a previous response, pass the response_id from that response.

    Args:
        query: Your complex question or problem
        response_id: Pass the response_id from a previous response to continue that conversation
        max_tokens: Maximum response length (default: 16384)

    Returns:
        Detailed answer with reasoning. Use the returned response_id to ask follow-up questions.
    """
    result = _create_request(
        input_content=query,
        previous_response_id=response_id,
        max_tokens=max_tokens,
        system_instruction="Think step by step. Be thorough and cite sources.",
        use_web_search=True,
        use_reasoning=True,
    )

    return _format_response(result)


@mcp.tool
def chat(
    query: str,
    response_id: Optional[str] = None,
    max_tokens: int = 8192,
) -> str:
    """
    Chat with Grok without web search.

    Use for casual conversation with Grok or creative tasks specific to the xAI ecosystem.
    To follow up on a previous response, pass the response_id from that response.

    Args:
        query: Your message
        response_id: Pass the response_id from a previous response to continue that conversation
        max_tokens: Maximum response length (default: 8192)

    Returns:
        Response text. Use the returned response_id to continue the conversation.
    """
    result = _create_request(
        input_content=query,
        previous_response_id=response_id,
        max_tokens=max_tokens,
        system_instruction=None,
        use_web_search=False,
    )

    return _format_response(result)


@mcp.tool
def x_search(
    query: str,
    max_results: int = 10,
    allowed_handles: Optional[list[str]] = None,
    excluded_handles: Optional[list[str]] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    enable_images: bool = False,
    enable_video: bool = False,
) -> str:
    """
    Search X (Twitter) posts using Grok. Returns structured results from X.

    Use this to find posts, discussions, and opinions from X users.

    Args:
        query: Search query
        max_results: Maximum number of results to return (default: 10)
        allowed_handles: Only include posts from these X handles (max 10, without @)
        excluded_handles: Exclude posts from these X handles (max 10, without @)
        from_date: Start date in YYYY-MM-DD format
        to_date: End date in YYYY-MM-DD format
        enable_images: Allow model to analyze images in posts
        enable_video: Allow model to analyze videos in posts

    Returns:
        Structured search results with post content, authors, and URLs
    """
    system_instruction = f"""Search X for the query and return results in this exact format:

---
AUTHOR: @[handle]
POST: [post content]
URL: [full x.com url]
---

Return up to {max_results} results. No additional commentary or analysis."""

    result = _create_request(
        input_content=query,
        system_instruction=system_instruction,
        max_tokens=4096,
        use_web_search=False,
        use_x_search=True,
        allowed_x_handles=allowed_handles,
        excluded_x_handles=excluded_handles,
        from_date=from_date,
        to_date=to_date,
        enable_image_understanding=enable_images,
        enable_video_understanding=enable_video,
    )

    return _format_response(result)


# TODO: Add x_think tool - deep reasoning with X search grounding (use_reasoning=True, use_x_search=True)


@mcp.tool
def x_ask(
    query: str,
    response_id: Optional[str] = None,
    max_tokens: int = 8192,
    allowed_handles: Optional[list[str]] = None,
    excluded_handles: Optional[list[str]] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    enable_images: bool = False,
    enable_video: bool = False,
) -> str:
    """
    Get grounded answers from Grok using X (Twitter) posts as sources.

    Model automatically searches X when needed for current discussions and opinions.
    Great for understanding public sentiment, trending topics, or what people are saying.

    Args:
        query: Your question
        response_id: Pass the response_id from a previous response to continue that conversation
        max_tokens: Maximum response length (default: 8192)
        allowed_handles: Only include posts from these X handles (max 10, without @)
        excluded_handles: Exclude posts from these X handles (max 10, without @)
        from_date: Start date in YYYY-MM-DD format
        to_date: End date in YYYY-MM-DD format
        enable_images: Allow model to analyze images in posts
        enable_video: Allow model to analyze videos in posts

    Returns:
        Answer with X post sources. Use the returned response_id to ask follow-up questions.
    """
    result = _create_request(
        input_content=query,
        previous_response_id=response_id,
        max_tokens=max_tokens,
        system_instruction="Be concise and factual. Cite X posts when referencing discussions or opinions.",
        use_web_search=False,
        use_x_search=True,
        allowed_x_handles=allowed_handles,
        excluded_x_handles=excluded_handles,
        from_date=from_date,
        to_date=to_date,
        enable_image_understanding=enable_images,
        enable_video_understanding=enable_video,
    )

    return _format_response(result)


if __name__ == "__main__":
    mcp.run()
