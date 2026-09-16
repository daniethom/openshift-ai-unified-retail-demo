"""RAG tools backed by the RAG MCP server."""

from __future__ import annotations

from typing import Any

from agents import mcp_client


class MCPRagRetriever:
    """Retrieve contextual documents via the RAG MCP server."""

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return await mcp_client.retrieve_documents(query, top_k=top_k)

    async def retrieve_documents(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return await self.retrieve(query, top_k=top_k)
