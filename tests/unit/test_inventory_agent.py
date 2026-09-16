import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.inventory_agent import InventoryAgent


@pytest.fixture
def inventory_agent():
    agent = InventoryAgent(
        mcp_servers={"analytics_server": MagicMock()}, data_store=MagicMock()
    )
    agent._query_stock = AsyncMock(
        return_value={
            "on_hand": 50,
            "allocated": 10,
            "available": 40,
            "incoming": 25,
        }
    )
    agent._get_product_info = AsyncMock(
        return_value={
            "product_id": "MF001",
            "name": "Classic Wool Trench Coat",
            "price": 3499.99,
        }
    )
    return agent


@pytest.mark.parametrize(
    "query, expected_type",
    [
        ("How much stock do we have for product X?", "stock_check"),
        ("Please optimize our inventory for winter coats.", "optimization"),
        ("Can you forecast demand for next month?", "forecast"),
        ("Generate a reorder list for the Cape Town store.", "reorder"),
        ("Give me an overview.", "general"),
    ],
)
def test_classify_inventory_query(inventory_agent, query, expected_type):
    query_type = inventory_agent._classify_query(query)
    assert query_type == expected_type


@pytest.mark.asyncio
async def test_handle_stock_check_logic(inventory_agent):
    query = "Check stock for MF001"
    context = {"product_id": "MF001", "location": "cape_town"}

    result = await inventory_agent._handle_stock_check(query, context)

    inventory_agent._query_stock.assert_awaited()
    assert result["availability"] in {"high", "medium", "low", "out_of_stock"}
    assert "stock_levels" in result
    assert "reorder_status" in result


@pytest.mark.asyncio
async def test_process_query_handles_forecast_request(inventory_agent, monkeypatch):
    query = "Forecast demand for shirts"
    context = {}

    mock_handler = AsyncMock(return_value={"forecast": []})
    monkeypatch.setattr(inventory_agent, "_handle_forecast", mock_handler)

    await inventory_agent.process_query(query, context)

    mock_handler.assert_awaited_once_with(query, context)


@pytest.mark.asyncio
async def test_process_query_error_handling(inventory_agent, monkeypatch):
    query = "This will fail"

    monkeypatch.setattr(
        inventory_agent,
        "_classify_query",
        MagicMock(side_effect=ValueError("Classification Failed")),
    )

    result = await inventory_agent.process_query(query, {})

    assert result["status"] == "error"
    assert result["error"] == "Classification Failed"
    assert result["metadata"]["agent"] == "InventoryAgent"
