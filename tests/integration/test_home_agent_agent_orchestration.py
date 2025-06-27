import pytest
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Add the project root to the path to allow importing the agents
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents.home_agent import HomeAgent
from agents.inventory_agent import InventoryAgent

# --- Pytest Fixtures ---

@pytest.fixture
def mock_inventory_agent():
    """
    Provides a real InventoryAgent instance but with its external
    dependencies mocked out. We will spy on its process_query method.
    """
    # Mock the dependencies of the InventoryAgent
    mock_data_store = MagicMock()
    mock_mcp_servers = {} # Not needed for this specific test
    
    # Create a real InventoryAgent instance
    agent = InventoryAgent(mcp_servers=mock_mcp_servers, data_store=mock_data_store)
    
    # Spy on the process_query method using AsyncMock. This allows us to
    # track calls to it while still executing its real (or simplified) logic.
    # For this test, we'll have it return a predictable dictionary.
    agent.process_query = AsyncMock(
        return_value={"result": "mock inventory data", "status": "success"}
    )
    
    return agent

@pytest.fixture
def home_agent_with_mock_crew(mock_inventory_agent):
    """
    Provides a real HomeAgent instance configured with a crew of mock
    specialized agents, including our spied-upon InventoryAgent.
    """
    agent_registry = {
        "InventoryAgent": mock_inventory_agent,
        # Other agents could be added here as simple mocks if needed
        "PricingAgent": MagicMock(),
        "CustomerAgent": MagicMock(),
        "TrendAgent": MagicMock(),
    }
    
    # The home agent's own dependencies can be simple mocks for this test
    home_mcp_servers = {}
    
    # Create a real HomeAgent instance with the mocked crew
    agent = HomeAgent(mcp_servers=home_mcp_servers, agent_registry=agent_registry)
    return agent

# --- Integration Test Case ---

@pytest.mark.asyncio
async def test_home_agent_orchestrates_simple_inventory_query(home_agent_with_mock_crew, mock_inventory_agent):
    """
    Tests the integration between HomeAgent and InventoryAgent.
    It verifies that the HomeAgent correctly analyzes a simple inventory query,
    delegates it to the InventoryAgent, and returns its response.
    """
    # Arrange: A simple query that should be routed to the InventoryAgent
    query = "check stock for product 123"
    context = {}
    
    # Act: Call the main processing method on the orchestrator
    final_result = await home_agent_with_mock_crew.process_query(query, context)
    
    # Assert
    # 1. Verify that the HomeAgent's routing logic worked and it called the InventoryAgent
    mock_inventory_agent.process_query.assert_awaited_once()

    # 2. Verify that the final response from the HomeAgent contains the data
    #    that was returned by our mock InventoryAgent. This proves the full
    #    communication loop worked.
    assert final_result["status"] == "success"
    # The home agent wraps the specialist agent's result in its own response structure
    agent_results = final_result["response"]["detailed_insights"]
    assert "InventoryAgent" in agent_results
    assert agent_results["InventoryAgent"]["result"] == "mock inventory data"
