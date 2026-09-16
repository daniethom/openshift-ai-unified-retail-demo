"""Tests for MCP-backed RAG retriever tool."""

import os
import sys
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.tools.rag_tools import MCPRagRetriever


@pytest.mark.asyncio
async def test_retrieve_returns_documents(monkeypatch):
    monkeypatch.setattr(
        "agents.tools.rag_tools.mcp_client.retrieve_documents",
        AsyncMock(return_value=[{"source": "doc.md", "content": "text", "score": 0.9}]),
    )

    retriever = MCPRagRetriever()
    docs = await retriever.retrieve("winter fashion", top_k=2)

    assert docs[0]["source"] == "doc.md"
