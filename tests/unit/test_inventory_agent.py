import pytest
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents.inventory_agent import InventoryAgent

# --- Pytest Fixtures ---

@pytest.fixture
def mock_data_store():
    """
    A fixture that provides a mock data_store object for the InventoryAgent.
    It simulates responses for data-fetching methods.
    """
    store = MagicMock()
    # Note: _query_stock is a private method in the agent, but it represents
    # the core data fetching. We can also mock the public-facing wrappers.
    store.query_stock = AsyncMock(return_value={
        "on_hand": 50,
        "allocated": 10,
        "available": 40,
        "incoming": 25,
    })
    store.get_product_info = AsyncMock(return_value={
        "product_id": "MF-BLZ-001", "name": "Classic Blazer", "price": 1999.00
    })
    return store

@pytest.fixture
def inventory_agent(mock_data_store):
    """
    Provides an instance of the InventoryAgent, initialized with the mock data_store.
    """
    mcp_servers = {"analytics_server": MagicMock()}
    agent = InventoryAgent(mcp_servers=mcp_servers, data_store=mock_data_store)
    # We can also mock the internal query method directly for more control
    agent._query_stock = AsyncMock(return_value={
        "on_hand": 50, "available": 40, "incoming": 25
    })
    return agent

# --- Test Cases ---

@pytest.mark.parametrize("query, expected_type", [
    ("How much stock do we have for product X?", "stock_check"),
    ("Please optimize our inventory for winter coats.", "optimization"),
    ("Can you forecast demand for next month?", "forecast"),
    ("Generate a reorder list for the Cape Town store.", "reorder"),
    ("Give me an overview.", "general")
])
def test_classify_inventory_query(inventory_agent, query, expected_type):
    """
    Tests the query classification logic for the InventoryAgent.
    """
    # Act
    query_type = inventory_agent._classify_query(query)
    
    # Assert
    assert query_type == expected_type

@pytest.mark.asyncio
async def test_handle_stock_check_logic(inventory_agent):
    """
    Tests the internal logic of the _handle_stock_check method to ensure
    it correctly processes data and calculates availability.
    """
    # Arrange
    query = "Check stock for MF-BLZ-001"
    context = {"product_id": "MF-BLZ-001"}
    
    # Act
    result = await inventory_agent._handle_stock_check(query, context)
    
    # Assert
    # Check that the mock method was called
    inventory_agent._query_stock.assert_called()
    
    # Check the calculated results based on mock data
    assert result["availability"] == "medium"
    assert result["stock_levels"]["cape_town"]["on_hand"] == 50 # Example location from default call
    assert result["reorder_status"]["needs_reorder"] is True # Based on simplified reorder logic

@pytest.mark.asyncio
async def test_process_query_handles_forecast_request(inventory_agent, monkeypatch):
    """
    Tests the main process_query router to ensure it correctly calls the
    forecast handler when a forecast-related query is made.
    """
    # Arrange
    query = "Forecast demand for shirts"
    context = {}
    
    # Mock the specific handler to confirm it gets called
    mock_handler = AsyncMock(return_value={"forecast": []})
    monkeypatch.setattr(inventory_agent, '_handle_forecast', mock_handler)
    
    # Act
    await inventory_agent.process_query(query, context)
    
    # Assert
    mock_handler.assert_awaited_once_with(query, context)

@pytest.mark.asyncio
async def test_process_query_error_handling(inventory_agent, monkeypatch):
    """
    Tests that if an internal method fails, the main process_query method
    catches the exception and returns a formatted error.
    """
    # Arrange
    query = "This will fail"
    
    # Mock the query classifier to raise an unexpected error
    monkeypatch.setattr(inventory_agent, '_classify_query', MagicMock(side_effect=ValueError("Classification Failed")))
    
    # Act
    result = await inventory_agent.process_query(query, {})
    
    # Assert
    assert result["status"] == "error"
    assert result["error"] == "Classification Failed"
    assert result["metadata"]["agent"] == "InventoryAgent"

