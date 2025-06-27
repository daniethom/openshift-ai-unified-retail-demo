import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents.customer_agent import CustomerAgent

# --- Mock Implementation for Testing ---

@pytest.fixture
def mock_data_store():
    """
    A fixture that provides a mock data_store object.
    We use AsyncMock for methods that the agent will 'await'.
    """
    store = MagicMock()
    store.get_customer_profile = AsyncMock(return_value={"customer_id": "CUST123", "tier": "gold"})
    store.get_recent_orders = AsyncMock(return_value=[{"order_id": "ORD567", "status": "shipped"}])
    # Add other mock methods as needed for more detailed tests
    return store

@pytest.fixture
def customer_agent(mock_data_store):
    """
    Provides an instance of the CustomerAgent, initialized with the mock data_store.
    """
    # The mcp_servers can be a simple dict for these tests
    mcp_servers = {"mock_server": {}}
    agent = CustomerAgent(mcp_servers=mcp_servers, data_store=mock_data_store)
    return agent

# --- Test Cases ---

@pytest.mark.parametrize("query, expected_type", [
    ("How can I track my order?", "support"),
    ("I need a suggestion for a new dress.", "recommendation"),
    ("I'm very unhappy with my last purchase.", "complaint"),
    ("How many loyalty points do I have?", "loyalty"),
    ("What are your store hours?", "general")
])
def test_classify_customer_query(customer_agent, query, expected_type):
    """
    Tests the query classification logic with various inputs.
    """
    # Act
    query_type = customer_agent._classify_customer_query(query)
    
    # Assert
    assert query_type == expected_type

@pytest.mark.asyncio
async def test_process_query_handles_support_request(customer_agent, mock_data_store):
    """
    Tests the end-to-end processing of a standard support query.
    It verifies that the correct internal helper methods are called.
    """
    # Arrange
    query = "Where is my order?"
    context = {"customer_id": "CUST123"}
    
    # Act
    result = await customer_agent.process_query(query, context)
    
    # Assert
    assert result["status"] == "success"
    assert result["query_type"] == "support"
    
    # Check that the data store was called to get customer information
    mock_data_store.get_customer_profile.assert_awaited_with("CUST123")
    mock_data_store.get_recent_orders.assert_awaited_with("CUST123")
    
    # Check the structure of the result
    assert "response" in result["result"]
    assert "relevant_information" in result["result"]
    assert "recent_orders" in result["result"]["relevant_information"]

@pytest.mark.asyncio
async def test_process_query_handles_recommendation_request(customer_agent, monkeypatch):
    """
    Tests the recommendation query pathway. We mock the internal handler to
    isolate the process_query routing logic.
    """
    # Arrange
    query = "I need a new pair of shoes."
    context = {"customer_id": "CUST123"}
    
    # Mock the specific handler to confirm it gets called
    mock_handler = AsyncMock(return_value={"recommendations": []})
    monkeypatch.setattr(customer_agent, '_handle_recommendation_query', mock_handler)
    
    # Act
    await customer_agent.process_query(query, context)
    
    # Assert
    mock_handler.assert_awaited_once_with(query, context)

@pytest.mark.asyncio
async def test_process_query_general_error_handling(customer_agent, monkeypatch):
    """
    Tests that if any step in the process_query pipeline fails,
    an error response is generated correctly.
    """
    # Arrange
    query = "This query will fail"
    context = {"customer_id": "CUST123"}
    
    # Mock the classification step to raise an exception
    monkeypatch.setattr(customer_agent, '_classify_customer_query', MagicMock(side_effect=Exception("Simulated failure")))
    
    # Act
    result = await customer_agent.process_query(query, context)
    
    # Assert
    assert result["status"] == "error"
    assert result["error"] == "Simulated failure"
