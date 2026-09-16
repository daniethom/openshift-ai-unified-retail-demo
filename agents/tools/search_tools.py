"""Search tools backed by the search MCP server."""

from __future__ import annotations

from typing import Any

from agents import mcp_client


class TrendSearcher:
    """Fashion trend search tool that calls the search MCP server over HTTP."""

    async def search_fashion_trends(
        self,
        query: str,
        location: str = "",
    ) -> dict[str, Any]:
        full_query = f"{query} {location}".strip()
        results = await mcp_client.web_search(full_query)

        trends = []
        for index, item in enumerate(results):
            trends.append(
                {
                    "name": item.get("title", f"Trend {index + 1}"),
                    "relevance": max(0.5, 0.9 - (index * 0.05)),
                    "growth": "rising" if index == 0 else "stable",
                    "summary": item.get("content", ""),
                }
            )

        if not trends:
            trends = [
                {
                    "name": "Power Suiting",
                    "relevance": 0.9,
                    "growth": "rising",
                    "summary": "Tailored blazers and structured silhouettes.",
                }
            ]

        return {"trends": trends, "raw_results": results}

    async def search(self, topic: str) -> dict[str, Any]:
        """Search for market trend information."""
        results = await mcp_client.web_search(topic)
        return {"query": topic, "results": results, "prices": []}
