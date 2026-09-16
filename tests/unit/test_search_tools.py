"""Tests for TrendSearcher MCP integration."""

import os
import sys
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.tools.search_tools import TrendSearcher


@pytest.mark.asyncio
async def test_search_fashion_trends_formats_results(monkeypatch):
    monkeypatch.setattr(
        "agents.tools.search_tools.mcp_client.web_search",
        AsyncMock(
            return_value=[
                {
                    "title": "Utility wear rises",
                    "content": "Gorpcore trend",
                    "url": "https://x.test",
                }
            ]
        ),
    )

    searcher = TrendSearcher()
    result = await searcher.search_fashion_trends("summer trends", location="cape_town")

    assert result["trends"]
    assert result["trends"][0]["name"] == "Utility wear rises"


@pytest.mark.asyncio
async def test_search_fashion_trends_uses_fallback_when_empty(monkeypatch):
    monkeypatch.setattr(
        "agents.tools.search_tools.mcp_client.web_search",
        AsyncMock(return_value=[]),
    )

    searcher = TrendSearcher()
    result = await searcher.search_fashion_trends("summer trends")

    assert result["trends"][0]["name"] == "Power Suiting"


@pytest.mark.asyncio
async def test_search_delegates_to_mcp_client(monkeypatch):
    monkeypatch.setattr(
        "agents.tools.search_tools.mcp_client.web_search",
        AsyncMock(return_value=[{"title": "Price check", "content": "data"}]),
    )

    searcher = TrendSearcher()
    result = await searcher.search("pricing trends")

    assert result["query"] == "pricing trends"
    assert len(result["results"]) == 1
