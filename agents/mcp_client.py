"""HTTP client for MCP server /invoke endpoints."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0


class MCPClientError(Exception):
    """Raised when an MCP server request fails."""


async def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        logger.warning("MCP request failed for %s: %s", url, exc)
        raise MCPClientError(str(exc)) from exc


async def invoke_tool(
    base_url: str,
    tool_name: str,
    input_data: dict[str, Any] | None = None,
) -> Any:
    """Call a tool-based MCP server (RAG, search, analytics)."""
    url = f"{base_url.rstrip('/')}/invoke"
    payload = {"tool_name": tool_name, "input_data": input_data or {}}
    data = await _post_json(url, payload)
    if isinstance(data, dict) and "result" in data:
        return data["result"]
    return data


async def invoke_llm(prompt: str) -> str:
    """Call the LLM MCP server."""
    url = f"{settings.llm_mcp_url.rstrip('/')}/invoke"
    data = await _post_json(url, {"prompt": prompt})
    if isinstance(data, dict):
        return str(data.get("response", ""))
    return str(data)


async def web_search(query: str) -> list[dict[str, Any]]:
    """Run a web search via the search MCP server."""
    try:
        result = await invoke_tool(
            settings.search_mcp_url, "web_search", {"query": query}
        )
    except MCPClientError:
        return []

    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        return result.get("results", [])
    return []


async def retrieve_documents(
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve documents via the RAG MCP server."""
    try:
        result = await invoke_tool(
            settings.rag_mcp_url,
            "retrieve_documents",
            {"query": query, "top_k": top_k},
        )
    except MCPClientError:
        return []

    if isinstance(result, list):
        return result
    return []


async def get_demand_analytics(product_id: str) -> dict[str, Any]:
    """Fetch demand analytics when supported by the analytics MCP server."""
    try:
        result = await invoke_tool(
            settings.analytics_mcp_url,
            "get_demand_analytics",
            {"product_id": product_id},
        )
    except MCPClientError:
        return {}

    return result if isinstance(result, dict) else {}


async def get_product_details(product_id: str) -> dict[str, Any]:
    """Fetch product details when supported by the analytics MCP server."""
    try:
        result = await invoke_tool(
            settings.analytics_mcp_url,
            "get_product_details",
            {"product_id": product_id},
        )
    except MCPClientError:
        return {}

    return result if isinstance(result, dict) else {}


async def check_mcp_health(base_url: str) -> bool:
    """Best-effort health check for an MCP server URL."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            for path in ("/healthz", "/health", "/docs"):
                response = await client.get(f"{base_url.rstrip('/')}{path}")
                if response.status_code < 500:
                    return True
    except httpx.HTTPError:
        return False
    return False
