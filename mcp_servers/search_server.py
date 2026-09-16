# mcp_servers/search_server.py

import logging
import os
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config.settings import settings

logger = logging.getLogger(__name__)
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8003"))


class ToolInput(BaseModel):
    tool_name: str
    input_data: Dict[str, Any]


class SearchResult(BaseModel):
    title: str
    url: str
    content: str


class ToolOutput(BaseModel):
    status: str = "success"
    result: List[SearchResult]


app = FastAPI(
    title="Search MCP Server",
    description="Provides standardized access to real-time web search (Tavily).",
    version="1.0.0",
)


def _fallback_search(query: str) -> List[Dict[str, str]]:
    return [
        {
            "title": f"Retail trends related to: {query}",
            "url": "https://example.com/fallback-result",
            "content": (
                "Fallback search result used because Tavily is unavailable. "
                "Configure TAVILY_API_KEY for live web search."
            ),
        }
    ]


def tavily_search(query: str, max_results: int | None = None) -> List[Dict[str, str]]:
    """Search the web using Tavily, with fallback when unavailable."""
    limit = max_results or settings.tavily_max_results

    if not settings.tavily_api_key:
        logger.warning("TAVILY_API_KEY is not set; returning fallback search results.")
        return _fallback_search(query)

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(query=query, max_results=limit)
        results = response.get("results", [])

        formatted = []
        for item in results:
            formatted.append(
                {
                    "title": item.get("title", "Untitled result"),
                    "url": item.get("url", ""),
                    "content": item.get("content", item.get("snippet", "")),
                }
            )
        return formatted or _fallback_search(query)
    except Exception as exc:
        logger.warning("Tavily search failed, using fallback: %s", exc)
        return _fallback_search(query)


@app.get("/healthz")
async def health_check() -> dict[str, str]:
    status = "configured" if settings.tavily_api_key else "fallback"
    return {"status": status}


@app.post("/invoke", response_model=ToolOutput)
async def invoke_tool(payload: ToolInput):
    if payload.tool_name == "web_search":
        query = payload.input_data.get("query")
        if not query:
            raise HTTPException(
                status_code=400, detail="Missing 'query' in input_data."
            )

        max_results = payload.input_data.get("max_results")
        search_results = tavily_search(
            query=query,
            max_results=int(max_results) if max_results else None,
        )
        return ToolOutput(result=search_results)

    raise HTTPException(
        status_code=404, detail=f"Tool '{payload.tool_name}' not found."
    )


if __name__ == "__main__":
    logger.info("Starting Search MCP Server on port %s", MCP_SERVER_PORT)
    uvicorn.run(app, host="0.0.0.0", port=MCP_SERVER_PORT)
