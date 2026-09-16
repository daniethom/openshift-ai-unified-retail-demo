import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.customer_agent import CustomerAgent


@pytest.fixture
def mock_data_store():
    store = MagicMock()
    store.get_customer_profile = AsyncMock(
        return_value={"customer_id": "CUST123", "tier": "gold"}
    )
    store.get_recent_orders = AsyncMock(
        return_value=[{"order_id": "ORD567", "status": "shipped"}]
    )
    return store


@pytest.fixture
def customer_agent(mock_data_store):
    return CustomerAgent(mcp_servers={"mock_server": {}}, data_store=mock_data_store)


@pytest.mark.parametrize(
    "query, expected_type",
    [
        ("How can I track my order?", "support"),
        ("I need a suggestion for a new dress.", "recommendation"),
        ("I'm very unhappy with my last purchase.", "complaint"),
        ("How many loyalty points do I have?", "loyalty"),
        ("What are your store hours?", "general"),
    ],
)
def test_classify_customer_query(customer_agent, query, expected_type):
    query_type = customer_agent._classify_customer_query(query)
    assert query_type == expected_type


@pytest.mark.asyncio
async def test_process_query_handles_support_request(customer_agent, monkeypatch):
    query = "Where is my order?"
    context = {"customer_id": "CUST123"}

    monkeypatch.setattr(
        customer_agent,
        "_get_customer_profile",
        AsyncMock(return_value={"customer_id": "CUST123", "tier": "gold"}),
    )
    monkeypatch.setattr(
        customer_agent,
        "_gather_relevant_info",
        AsyncMock(return_value={"recent_orders": [{"order_id": "ORD567"}]}),
    )

    result = await customer_agent.process_query(query, context)

    assert result["status"] == "success"
    assert result["query_type"] == "support"
    assert "response" in result["result"]
    assert "recent_orders" in result["result"]["relevant_information"]


@pytest.mark.asyncio
async def test_process_query_handles_recommendation_request(
    customer_agent, monkeypatch
):
    query = "I need a new pair of shoes."
    context = {"customer_id": "CUST123"}

    mock_handler = AsyncMock(return_value={"recommendations": []})
    monkeypatch.setattr(customer_agent, "_handle_recommendation_query", mock_handler)

    await customer_agent.process_query(query, context)

    mock_handler.assert_awaited_once_with(query, context)


@pytest.mark.asyncio
async def test_process_query_general_error_handling(customer_agent, monkeypatch):
    query = "This query will fail"
    context = {"customer_id": "CUST123"}

    monkeypatch.setattr(
        customer_agent,
        "_classify_customer_query",
        MagicMock(side_effect=Exception("Simulated failure")),
    )

    result = await customer_agent.process_query(query, context)

    assert result["status"] == "error"
    assert result["error"] == "Simulated failure"
