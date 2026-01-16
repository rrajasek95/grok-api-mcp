"""
Pytest configuration for Grok API MCP Server tests.

This conftest.py adds the parent directory to the path so that
the server module can be imported properly.
"""

import os
import sys

# Add parent directory to path for server module import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set required environment variable before any test imports
os.environ.setdefault("XAI_API_KEY", "test_api_key")
