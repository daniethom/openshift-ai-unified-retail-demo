import pytest
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Add the project root to the path to allow importing the agents
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents.home_agent import HomeAgent
from agents.inventory_agent import InventoryAgent
from agents.pricing_agent import PricingAgent

# --- Pytest Fixtures ---

@pytest.fixture
def mock_inventory_agent():
    """
    Provides a real InventoryAgent instance but with its external
    dependencies mocked out. We will spy on its process_query method.
    """
    agent = InventoryAgent(mcp_servers={}, data_store=MagicMock())
    # Spy on the process_query method using AsyncMock
    agent.process_query = AsyncMock(
        return_value={"result": "mock inventory data", "status": "success"}
    )
    return agent

@pytest.fixture
def mock_pricing_agent():
    """
    Provides a real PricingAgent instance with mocked dependencies,
    similar to the mock_inventory_agent.
    """
    agent = PricingAgent(mcp_servers={}, data_store=MagicMock())
    # Spy on the process_query method with a unique return value
    agent.process_query = AsyncMock(
        return_value={"result": "mock pricing data", "status": "success"}
    )
    return agent

@pytest.fixture
def home_agent_with_mock_crew(mock_inventory_agent, mock_pricing_agent):
    """
    Provides a real HomeAgent instance configured with a crew of spied-upon
    specialized agents.
    """
    agent_registry = {
        "InventoryAgent": mock_inventory_agent,
        "PricingAgent": mock_pricing_agent,
        # Other agents can be simple mocks if not part of the test
        "CustomerAgent": MagicMock(),
        "TrendAgent": MagicMock(),
    }
    
    # Create a real HomeAgent instance with the mocked crew
    agent = HomeAgent(mcp_servers={}, agent_registry=agent_registry)
    return agent

# --- Integration Test Cases ---

@pytest.mark.asyncio
async def test_home_agent_orchestrates_simple_inventory_query(home_agent_with_mock_crew, mock_inventory_agent):
    """
    Tests the integration between HomeAgent and a single specialized agent.
    It verifies that the HomeAgent correctly routes a simple query.
    """
    # Arrange: A simple query that should be routed to the InventoryAgent
    query = "check stock for product 123"
    context = {}
    
    # Act: Call the main processing method on the orchestrator
    final_result = await home_agent_with_mock_crew.process_query(query, context)
    
    # Assert
    # 1. Verify the correct agent was called
    mock_inventory_agent.process_query.assert_awaited_once()

    # 2. Verify the response contains the data from the specialized agent
    assert final_result["status"] == "success"
    agent_results = final_result["response"]["detailed_insights"]
    assert "InventoryAgent" in agent_results
    assert agent_results["InventoryAgent"]["result"] == "mock inventory data"

@pytest.mark.asyncio
async def test_home_agent_orchestrates_parallel_query(home_agent_with_mock_crew, mock_inventory_agent, mock_pricing_agent):
    """
    Tests the HomeAgent's ability to orchestrate a query requiring parallel
    execution of multiple agents (Inventory and Pricing).
    """
    # Arrange: A query with keywords that trigger both agents
    query = "What is the price and stock level for product XYZ?"
    context = {}

    # Act: Call the orchestrator
    final_result = await home_agent_with_mock_crew.process_query(query, context)

    # Assert
    # 1. Verify that BOTH specialized agents were called exactly once
    mock_inventory_agent.process_query.assert_awaited_once()
    mock_pricing_agent.process_query.assert_awaited_once()

    # 2. Verify that the final response contains the synthesized results
    #    from both agents.
    assert final_result["status"] == "success"
    agent_results = final_result["response"]["detailed_insights"]
    assert "InventoryAgent" in agent_results
    assert "PricingAgent" in agent_results
    assert agent_results["InventoryAgent"]["result"] == "mock inventory data"
    assert agent_results["PricingAgent"]["result"] == "mock pricing data"
