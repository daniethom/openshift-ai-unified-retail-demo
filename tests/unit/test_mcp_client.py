"""Unit tests for the MCP HTTP client."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents import mcp_client


@pytest.mark.asyncio
async def test_web_search_returns_list(monkeypatch):
    monkeypatch.setattr(
        mcp_client,
        "invoke_tool",
        AsyncMock(return_value=[{"title": "Trend A", "content": "Details"}]),
    )

    results = await mcp_client.web_search("winter fashion")

    assert len(results) == 1
    assert results[0]["title"] == "Trend A"


@pytest.mark.asyncio
async def test_web_search_returns_empty_on_error(monkeypatch):
    monkeypatch.setattr(
        mcp_client,
        "invoke_tool",
        AsyncMock(side_effect=mcp_client.MCPClientError("connection failed")),
    )

    results = await mcp_client.web_search("winter fashion")

    assert results == []


@pytest.mark.asyncio
async def test_invoke_llm_returns_response(monkeypatch):
    monkeypatch.setattr(
        mcp_client,
        "_post_json",
        AsyncMock(return_value={"response": "Hello from LLM"}),
    )

    response = await mcp_client.invoke_llm("Say hello")

    assert response == "Hello from LLM"
