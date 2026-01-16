"""
Tests for Grok API MCP Server

Uses pytest with respx for HTTP mocking.
Run with: uv run pytest
"""

import json

import respx
from httpx import Response

# Import server module (conftest.py handles path and env setup)
from server import (
    _parse_response,
    _format_response,
    _create_request,
    API_ENDPOINT,
)


class TestParseResponse:
    """Tests for _parse_response function."""

    def test_parse_simple_message(self):
        """Parse a simple message response."""
        data = {
            "id": "resp_123",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Hello, world!"}
                    ]
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }

        result = _parse_response(data)

        assert result["response_id"] == "resp_123"
        assert result["status"] == "completed"
        assert result["text"] == "Hello, world!"
        assert result["sources"] == []
        assert result["usage"]["input_tokens"] == 10

    def test_parse_message_with_annotations(self):
        """Parse message with URL annotations."""
        data = {
            "id": "resp_456",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "text",
                            "text": "Check this source.",
                            "annotations": [
                                {"url": "https://example.com", "title": "Example"}
                            ]
                        }
                    ]
                }
            ]
        }

        result = _parse_response(data)

        assert result["text"] == "Check this source."
        assert len(result["sources"]) == 1
        assert result["sources"][0]["url"] == "https://example.com"
        assert result["sources"][0]["title"] == "Example"

    def test_parse_web_search_result(self):
        """Parse web search results."""
        data = {
            "id": "resp_789",
            "status": "completed",
            "output": [
                {
                    "type": "web_search_result",
                    "results": [
                        {"url": "https://news.com/article", "title": "News Article"},
                        {"url": "https://blog.com/post", "title": "Blog Post"}
                    ]
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Found results."}]
                }
            ]
        }

        result = _parse_response(data)

        assert result["text"] == "Found results."
        assert len(result["sources"]) == 2
        assert result["sources"][0]["title"] == "News Article"
        assert result["sources"][1]["title"] == "Blog Post"

    def test_parse_x_search_result(self):
        """Parse X search results."""
        data = {
            "id": "resp_x123",
            "status": "completed",
            "output": [
                {
                    "type": "x_search_result",
                    "results": [
                        {"url": "https://x.com/user/status/123", "title": "@user"},
                        {"url": "https://x.com/other/status/456", "author": "other"}
                    ]
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Found X posts."}]
                }
            ]
        }

        result = _parse_response(data)

        assert result["text"] == "Found X posts."
        assert len(result["sources"]) == 2
        assert result["sources"][0]["title"] == "@user"
        assert result["sources"][1]["title"] == "other"  # Falls back to author

    def test_parse_sources_from_different_outputs(self):
        """Sources from multiple outputs are collected."""
        data = {
            "id": "resp_multi",
            "status": "completed",
            "output": [
                {
                    "type": "web_search_result",
                    "results": [
                        {"url": "https://example1.com", "title": "First"}
                    ]
                },
                {
                    "type": "web_search_result",
                    "results": [
                        {"url": "https://example2.com", "title": "Second"}
                    ]
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Results."}]
                }
            ]
        }

        result = _parse_response(data)

        assert len(result["sources"]) == 2

    def test_parse_empty_response(self):
        """Handle empty or minimal response."""
        data = {"id": "resp_empty", "output": []}

        result = _parse_response(data)

        assert result["response_id"] == "resp_empty"
        assert result["text"] == ""
        assert result["sources"] == []


class TestFormatResponse:
    """Tests for _format_response function."""

    def test_format_simple_response(self):
        """Format a simple response with text only."""
        result = {
            "text": "Hello, world!",
            "sources": [],
            "response_id": "resp_123"
        }

        output = _format_response(result)

        assert "Hello, world!" in output
        assert "response_id: resp_123" in output
        assert "Sources:" not in output

    def test_format_response_with_sources(self):
        """Format response with sources."""
        result = {
            "text": "Answer text.",
            "sources": [
                {"title": "Source 1", "url": "https://example1.com"},
                {"title": "Source 2", "url": "https://example2.com"}
            ],
            "response_id": "resp_456"
        }

        output = _format_response(result)

        assert "Answer text." in output
        assert "Sources:" in output
        assert "[Source 1](https://example1.com)" in output
        assert "[Source 2](https://example2.com)" in output

    def test_format_error_response(self):
        """Format an error response."""
        result = {
            "error": "API rate limit exceeded",
            "response_id": None,
            "status": "failed"
        }

        output = _format_response(result)

        assert "Error: API rate limit exceeded" in output

    def test_format_response_no_id(self):
        """Format response without response_id."""
        result = {
            "text": "Text",
            "sources": [],
            "response_id": None
        }

        output = _format_response(result)

        assert "Text" in output
        assert "response_id:" not in output


class TestCreateRequest:
    """Tests for _create_request function with mocked HTTP."""

    @respx.mock
    def test_web_search_request(self):
        """Test request with web search enabled."""
        mock_response = {
            "id": "resp_web",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Web result"}]
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20}
        }

        route = respx.post(API_ENDPOINT).mock(
            return_value=Response(200, json=mock_response)
        )

        result = _create_request(
            input_content="test query",
            use_web_search=True,
            use_x_search=False
        )

        assert route.called
        request = route.calls[0].request
        body = json.loads(request.content.decode())

        # Check tools contains web_search
        assert any(t.get("type") == "web_search" for t in body.get("tools", []))
        assert not any(t.get("type") == "x_search" for t in body.get("tools", []))
        assert result["text"] == "Web result"

    @respx.mock
    def test_x_search_request(self):
        """Test request with X search enabled."""
        mock_response = {
            "id": "resp_x",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "X result"}]
                }
            ]
        }

        route = respx.post(API_ENDPOINT).mock(
            return_value=Response(200, json=mock_response)
        )

        result = _create_request(
            input_content="test query",
            use_web_search=False,
            use_x_search=True
        )

        assert route.called
        request = route.calls[0].request
        body = json.loads(request.content.decode())

        assert any(t.get("type") == "x_search" for t in body.get("tools", []))
        assert not any(t.get("type") == "web_search" for t in body.get("tools", []))
        assert result["text"] == "X result"

    @respx.mock
    def test_x_search_with_filters(self):
        """Test X search with all filter options."""
        mock_response = {
            "id": "resp_filtered",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Filtered"}]
                }
            ]
        }

        route = respx.post(API_ENDPOINT).mock(
            return_value=Response(200, json=mock_response)
        )

        _create_request(
            input_content="test",
            use_web_search=False,
            use_x_search=True,
            allowed_x_handles=["user1", "user2"],
            excluded_x_handles=None,
            from_date="2025-01-01",
            to_date="2025-01-15",
            enable_image_understanding=True,
            enable_video_understanding=True
        )

        assert route.called
        request = route.calls[0].request
        body = json.loads(request.content.decode())

        x_search_tool = next(t for t in body["tools"] if t["type"] == "x_search")
        assert x_search_tool["allowed_x_handles"] == ["user1", "user2"]
        assert x_search_tool["from_date"] == "2025-01-01"
        assert x_search_tool["to_date"] == "2025-01-15"
        assert x_search_tool["enable_image_understanding"] is True
        assert x_search_tool["enable_video_understanding"] is True

    @respx.mock
    def test_reasoning_model(self):
        """Test that reasoning model is used when specified."""
        mock_response = {
            "id": "resp_think",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Thought"}]
                }
            ]
        }

        route = respx.post(API_ENDPOINT).mock(
            return_value=Response(200, json=mock_response)
        )

        _create_request(
            input_content="think about this",
            use_reasoning=True
        )

        request = route.calls[0].request
        body = json.loads(request.content.decode())

        assert body["model"] == "grok-4-1-fast"

    @respx.mock
    def test_non_reasoning_model(self):
        """Test that non-reasoning model is used by default."""
        mock_response = {
            "id": "resp_fast",
            "status": "completed",
            "output": []
        }

        route = respx.post(API_ENDPOINT).mock(
            return_value=Response(200, json=mock_response)
        )

        _create_request(
            input_content="quick question",
            use_reasoning=False
        )

        request = route.calls[0].request
        body = json.loads(request.content.decode())

        assert body["model"] == "grok-4-1-fast-non-reasoning"

    @respx.mock
    def test_system_instruction(self):
        """Test that system instruction is included."""
        mock_response = {"id": "resp_sys", "status": "completed", "output": []}

        route = respx.post(API_ENDPOINT).mock(
            return_value=Response(200, json=mock_response)
        )

        _create_request(
            input_content="query",
            system_instruction="Be concise."
        )

        request = route.calls[0].request
        body = json.loads(request.content.decode())

        system_msg = next((m for m in body["input"] if m["role"] == "system"), None)
        assert system_msg is not None
        assert system_msg["content"] == "Be concise."

    @respx.mock
    def test_previous_response_id(self):
        """Test conversation continuity with response_id."""
        mock_response = {"id": "resp_cont", "status": "completed", "output": []}

        route = respx.post(API_ENDPOINT).mock(
            return_value=Response(200, json=mock_response)
        )

        _create_request(
            input_content="follow up",
            previous_response_id="resp_previous"
        )

        request = route.calls[0].request
        body = json.loads(request.content.decode())

        assert body["previous_response_id"] == "resp_previous"

    @respx.mock
    def test_api_error_handling(self):
        """Test handling of API errors."""
        respx.post(API_ENDPOINT).mock(
            return_value=Response(429, json={"error": "Rate limited"})
        )

        result = _create_request(input_content="test")

        assert "error" in result
        assert "429" in result["error"]
        assert result["status"] == "failed"

    @respx.mock
    def test_no_tools_for_chat(self):
        """Test that no tools are included when both searches are disabled."""
        mock_response = {"id": "resp_chat", "status": "completed", "output": []}

        route = respx.post(API_ENDPOINT).mock(
            return_value=Response(200, json=mock_response)
        )

        _create_request(
            input_content="just chat",
            use_web_search=False,
            use_x_search=False
        )

        request = route.calls[0].request
        body = json.loads(request.content.decode())

        assert "tools" not in body or body.get("tools") == []

    @respx.mock
    def test_max_handles_limit(self):
        """Test that allowed_x_handles is limited to 10."""
        mock_response = {"id": "resp_limit", "status": "completed", "output": []}

        route = respx.post(API_ENDPOINT).mock(
            return_value=Response(200, json=mock_response)
        )

        handles = [f"user{i}" for i in range(15)]  # 15 handles

        _create_request(
            input_content="test",
            use_x_search=True,
            allowed_x_handles=handles
        )

        request = route.calls[0].request
        body = json.loads(request.content.decode())

        x_search_tool = next(t for t in body["tools"] if t["type"] == "x_search")
        assert len(x_search_tool["allowed_x_handles"]) == 10
        assert "user9" in x_search_tool["allowed_x_handles"]
        assert "user10" not in x_search_tool["allowed_x_handles"]

    @respx.mock
    def test_both_searches_enabled(self):
        """Test that both web and X search can be enabled together."""
        mock_response = {"id": "resp_both", "status": "completed", "output": []}

        route = respx.post(API_ENDPOINT).mock(
            return_value=Response(200, json=mock_response)
        )

        _create_request(
            input_content="search both",
            use_web_search=True,
            use_x_search=True
        )

        request = route.calls[0].request
        body = json.loads(request.content.decode())

        tool_types = [t["type"] for t in body["tools"]]
        assert "web_search" in tool_types
        assert "x_search" in tool_types


class TestIntegration:
    """Integration tests using internal functions directly."""

    @respx.mock
    def test_search_flow(self):
        """Test the search flow end-to-end."""
        mock_response = {
            "id": "resp_search",
            "status": "completed",
            "output": [
                {
                    "type": "web_search_result",
                    "results": [{"url": "https://news.com", "title": "News"}]
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Search results."}]
                }
            ]
        }

        respx.post(API_ENDPOINT).mock(
            return_value=Response(200, json=mock_response)
        )

        # Simulate what search tool does
        result = _create_request(
            input_content="test query",
            system_instruction="Return search results.",
            max_tokens=4096,
            use_web_search=True,
        )

        formatted = _format_response(result)

        assert "Search results." in formatted
        assert "[News](https://news.com)" in formatted

    @respx.mock
    def test_x_search_flow(self):
        """Test the X search flow end-to-end."""
        mock_response = {
            "id": "resp_xsearch",
            "status": "completed",
            "output": [
                {
                    "type": "x_search_result",
                    "results": [{"url": "https://x.com/u/status/1", "title": "@user"}]
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "X posts."}]
                }
            ]
        }

        respx.post(API_ENDPOINT).mock(
            return_value=Response(200, json=mock_response)
        )

        # Simulate what x_search tool does
        result = _create_request(
            input_content="test",
            max_tokens=4096,
            use_web_search=False,
            use_x_search=True,
            allowed_x_handles=["user"],
            from_date="2025-01-01"
        )

        formatted = _format_response(result)

        assert "X posts." in formatted
        assert "[@user](https://x.com/u/status/1)" in formatted

    @respx.mock
    def test_x_ask_flow(self):
        """Test the X ask flow end-to-end."""
        mock_response = {
            "id": "resp_xask",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Answer from X."}]
                }
            ]
        }

        respx.post(API_ENDPOINT).mock(
            return_value=Response(200, json=mock_response)
        )

        # Simulate what x_ask tool does
        result = _create_request(
            input_content="What are people saying?",
            max_tokens=8192,
            use_web_search=False,
            use_x_search=True,
            enable_image_understanding=True,
            enable_video_understanding=True
        )

        formatted = _format_response(result)

        assert "Answer from X." in formatted
        assert "response_id: resp_xask" in formatted

    @respx.mock
    def test_think_flow(self):
        """Test the think flow with reasoning model."""
        mock_response = {
            "id": "resp_think",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Deep analysis."}]
                }
            ]
        }

        route = respx.post(API_ENDPOINT).mock(
            return_value=Response(200, json=mock_response)
        )

        result = _create_request(
            input_content="Analyze this",
            max_tokens=16384,
            use_web_search=True,
            use_reasoning=True
        )

        # Verify reasoning model was used
        request = route.calls[0].request
        body = json.loads(request.content.decode())
        assert body["model"] == "grok-4-1-fast"

        formatted = _format_response(result)
        assert "Deep analysis." in formatted

    @respx.mock
    def test_chat_flow_no_search(self):
        """Test chat flow without any search."""
        mock_response = {
            "id": "resp_chat",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Just chatting."}]
                }
            ]
        }

        route = respx.post(API_ENDPOINT).mock(
            return_value=Response(200, json=mock_response)
        )

        result = _create_request(
            input_content="Hello",
            use_web_search=False,
            use_x_search=False
        )

        # Verify no tools
        request = route.calls[0].request
        body = json.loads(request.content.decode())
        assert "tools" not in body or body.get("tools") == []

        formatted = _format_response(result)
        assert "Just chatting." in formatted
