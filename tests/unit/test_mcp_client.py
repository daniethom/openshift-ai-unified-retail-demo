import os
import sys
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents import mcp_client


@pytest.mark.asyncio
async def test_invoke_tool_returns_result(monkeypatch):
    monkeypatch.setattr(
        mcp_client,
        "_post_json",
        AsyncMock(return_value={"status": "success", "result": {"value": 42}}),
    )

    result = await mcp_client.invoke_tool(
        "http://analytics:8004", "get_total_inventory_value"
    )

    assert result == {"value": 42}


@pytest.mark.asyncio
async def test_invoke_tool_raises_mcp_client_error(monkeypatch):
    monkeypatch.setattr(
        mcp_client,
        "_post_json",
        AsyncMock(side_effect=mcp_client.MCPClientError("boom")),
    )

    with pytest.raises(mcp_client.MCPClientError):
        await mcp_client.invoke_tool("http://analytics:8004", "missing")


@pytest.mark.asyncio
async def test_get_customer_profile_returns_dict(monkeypatch):
    monkeypatch.setattr(
        mcp_client,
        "invoke_tool",
        AsyncMock(return_value={"customer_id": "CUST001", "first_name": "Sarah"}),
    )

    profile = await mcp_client.get_customer_profile("CUST001")

    assert profile["customer_id"] == "CUST001"


@pytest.mark.asyncio
async def test_get_product_details_returns_empty_on_error(monkeypatch):
    monkeypatch.setattr(
        mcp_client,
        "invoke_tool",
        AsyncMock(side_effect=mcp_client.MCPClientError("down")),
    )

    product = await mcp_client.get_product_details("MF001")

    assert product == {}


@pytest.mark.asyncio
async def test_retrieve_documents_returns_list(monkeypatch):
    monkeypatch.setattr(
        mcp_client,
        "invoke_tool",
        AsyncMock(return_value=[{"source": "doc.md", "content": "text", "score": 0.9}]),
    )

    docs = await mcp_client.retrieve_documents("winter coats", top_k=3)

    assert len(docs) == 1


@pytest.mark.asyncio
async def test_get_demand_analytics_returns_dict(monkeypatch):
    monkeypatch.setattr(
        mcp_client,
        "invoke_tool",
        AsyncMock(return_value={"product_id": "MF001", "current_demand": "high"}),
    )

    analytics = await mcp_client.get_demand_analytics("MF001")

    assert analytics["product_id"] == "MF001"


@pytest.mark.asyncio
async def test_search_customers_by_name_returns_list(monkeypatch):
    monkeypatch.setattr(
        mcp_client,
        "invoke_tool",
        AsyncMock(return_value=[{"customer_id": "CUST001"}]),
    )

    customers = await mcp_client.search_customers_by_name("Sarah")

    assert customers[0]["customer_id"] == "CUST001"


@pytest.mark.asyncio
async def test_web_search_returns_list(monkeypatch):
    monkeypatch.setattr(
        mcp_client,
        "invoke_tool",
        AsyncMock(return_value=[{"title": "Trend A", "content": "Details"}]),
    )

    results = await mcp_client.web_search("winter fashion")

    assert len(results) == 1


@pytest.mark.asyncio
async def test_check_mcp_health_success(monkeypatch):
    response = AsyncMock(status_code=200)

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url):
            return response

    monkeypatch.setattr(mcp_client.httpx, "AsyncClient", lambda **kwargs: FakeClient())

    assert await mcp_client.check_mcp_health("http://analytics:8004") is True
