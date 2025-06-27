import pytest
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents.pricing_agent import PricingAgent

# --- Pytest Fixtures ---

@pytest.fixture
def mock_data_store():
    """Provides a mock for the data_store dependency."""
    return MagicMock()

@pytest.fixture
def mock_mcp_servers():
    """Provides a dictionary of mock MCP servers."""
    search_server = MagicMock()
    search_server.search = AsyncMock(return_value={"prices": []})

    analytics_server = MagicMock()
    analytics_server.get_demand_analytics = AsyncMock(return_value={"trend": "stable"})

    return {
        "search_server": search_server,
        "analytics_server": analytics_server,
    }

@pytest.fixture
def pricing_agent(mock_data_store, mock_mcp_servers):
    """Provides an instance of the PricingAgent with mocked dependencies."""
    agent = PricingAgent(mcp_servers=mock_mcp_servers, data_store=mock_data_store)
    # Also mock the internal data fetching methods to isolate the logic
    agent._get_current_pricing = AsyncMock(return_value={"product_id": "P1", "current_price": 100, "cost": 60})
    agent._get_market_data = AsyncMock(return_value={"market_average": 110, "price_range": {"min": 90, "max": 120}})
    agent._get_demand_data = AsyncMock(return_value={"current_demand": "stable", "price_sensitivity": -1.5})
    return agent

# --- Test Cases ---

@pytest.mark.parametrize("query, expected_type", [
    ("what is the best price for product P1?", "optimization"),
    ("analyze competitor pricing for shoes", "competition"),
    ("create a sale for our winter collection", "promotion"),
    ("what is the profit margin on blazers?", "margin"),
    ("tell me about our pricing", "general")
])
def test_classify_pricing_query(pricing_agent, query, expected_type):
    """
    Tests the query classification logic for various pricing-related queries.
    """
    # Act
    query_type = pricing_agent._classify_pricing_query(query)
    
    # Assert
    assert query_type == expected_type

@pytest.mark.asyncio
async def test_process_query_routes_to_competition_handler(pricing_agent, monkeypatch):
    """
    Tests that the main process_query method correctly routes a competition-related
    query to its specific handler.
    """
    # Arrange
    query = "How do we compare to competitor X?"
    context = {}
    
    # Mock the specific handler to confirm it gets called
    mock_handler = AsyncMock(return_value={"current_positioning": "competitive"})
    monkeypatch.setattr(pricing_agent, '_handle_competition_query', mock_handler)
    
    # Act
    result = await pricing_agent.process_query(query, context)
    
    # Assert
    mock_handler.assert_awaited_once_with(query, context)
    assert result["status"] == "success"
    assert result["query_type"] == "competition"

@pytest.mark.asyncio
async def test_handle_optimization_query_calls_dependencies(pricing_agent):
    """
    Tests the internal logic of the optimization handler to ensure it calls
    its dependencies (internal data fetching methods) correctly.
    """
    # Arrange
    query = "Optimize P1"
    context = {"product_id": "P1"}
    
    # Act
    result = await pricing_agent._handle_optimization_query(query, context)
    
    # Assert
    # Check that the data fetching methods were awaited
    pricing_agent._get_current_pricing.assert_awaited_once_with("P1")
    pricing_agent._get_market_data.assert_awaited_once_with("P1")
    pricing_agent._get_demand_data.assert_awaited_once_with("P1")
    
    # Check the result structure
    assert "optimization" in result
    assert "recommended_price" in result["optimization"]
    assert "expected_impact" in result

@pytest.mark.asyncio
async def test_process_query_general_error_handling(pricing_agent, monkeypatch):
    """
    Tests that if a dependency raises an exception during query processing,
    the main method catches it and returns a proper error response.
    """
    # Arrange
    query = "This query will cause an error"
    # Make one of the mocked methods raise an error
    pricing_agent._get_current_pricing = AsyncMock(side_effect=IOError("Database connection failed"))
    
    # Act
    result = await pricing_agent.process_query(query, {"product_id": "P1"})
    
    # Assert
    assert result["status"] == "error"
    assert "Database connection failed" in result["error"]
