import pytest
from unittest.mock import MagicMock, AsyncMock
import sys
import os
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents.home_agent import HomeAgent, QueryComplexity

# --- Pytest Fixtures ---

@pytest.fixture
def mock_specialized_agents():
    """
    Creates mock objects for each of the specialized agents.
    We use AsyncMock for the process_query method because it's an async function.
    """
    mock_inventory = MagicMock()
    mock_inventory.process_query = AsyncMock(return_value={"result": "inventory_data"})
    mock_inventory.name = "InventoryAgent"

    mock_pricing = MagicMock()
    mock_pricing.process_query = AsyncMock(return_value={"result": "pricing_data"})
    mock_pricing.name = "PricingAgent"

    mock_customer = MagicMock()
    mock_customer.process_query = AsyncMock(return_value={"result": "customer_data"})
    mock_customer.name = "CustomerAgent"
    
    mock_trend = MagicMock()
    mock_trend.process_query = AsyncMock(return_value={"result": "trend_data"})
    mock_trend.name = "TrendAgent"

    return {
        "InventoryAgent": mock_inventory,
        "PricingAgent": mock_pricing,
        "CustomerAgent": mock_customer,
        "TrendAgent": mock_trend,
    }

@pytest.fixture
def home_agent(mock_specialized_agents):
    """
    Provides an instance of the HomeAgent, initialized with our mock specialized agents.
    """
    # The mcp_servers can be a simple dict for these tests
    mcp_servers = {"mock_server": {}}
    
    agent = HomeAgent(mcp_servers=mcp_servers, agent_registry=mock_specialized_agents)
    return agent

# --- Test Cases ---

@pytest.mark.asyncio
async def test_analyze_query_simple(home_agent):
    """
    Tests that a simple informational query is analyzed correctly.
    """
    # Arrange
    query = "What is the stock level of product MF001?"
    
    # Act
    analysis = await home_agent._analyze_query(query, {})
    
    # Assert
    assert analysis["complexity"] == QueryComplexity.SIMPLE
    assert analysis["required_agents"] == ["InventoryAgent"]
    assert analysis["strategy"] == "direct"
    assert "stock" in analysis["keywords"]

@pytest.mark.asyncio
async def test_analyze_query_complex_analytical(home_agent):
    """
    Tests that a complex analytical query is analyzed correctly,
    requiring multiple agents and a hierarchical strategy.
    """
    # Arrange
    query = "Analyze pricing for winter coats based on current stock and market trends."
    
    # Act
    analysis = await home_agent._analyze_query(query, {})
    
    # Assert
    assert analysis["complexity"] == QueryComplexity.COMPLEX
    # The set comparison ignores order, which is good for this list.
    assert set(analysis["required_agents"]) == {"PricingAgent", "InventoryAgent", "TrendAgent"}
    assert analysis["strategy"] == "hierarchical"
    assert analysis["intent"]["question_type"] == "analytical"

@pytest.mark.asyncio
async def test_process_simple_query_orchestration(home_agent, mock_specialized_agents, monkeypatch):
    """
    Tests the end-to-end orchestration for a simple query.
    It mocks the analysis step to force a simple path and verifies that the
    correct agent is called.
    """
    # Arrange
    query = "stock check"
    mock_analysis_result = {
        "complexity": QueryComplexity.SIMPLE,
        "required_agents": ["InventoryAgent"],
        "strategy": "direct",
        "context_enhanced": {}
    }
    # We mock _analyze_query to isolate the orchestration logic of process_query
    monkeypatch.setattr(home_agent, '_analyze_query', AsyncMock(return_value=mock_analysis_result))

    # Act
    result = await home_agent.process_query(query, {})

    # Assert
    # Check that the InventoryAgent's process_query was called once
    mock_specialized_agents["InventoryAgent"].process_query.assert_awaited_once()
    # Check that other agents were NOT called
    mock_specialized_agents["PricingAgent"].process_query.assert_not_awaited()
    
    assert result["status"] == "success"
    assert "InventoryAgent" in result["response"]["detailed_insights"]

@pytest.mark.asyncio
async def test_process_moderate_parallel_query(home_agent, mock_specialized_agents, monkeypatch):
    """
    Tests the orchestration for a moderate query that should run in parallel.
    """
    # Arrange
    query = "What is the price and stock for our main products?"
    mock_analysis_result = {
        "complexity": QueryComplexity.MODERATE,
        "required_agents": ["PricingAgent", "InventoryAgent"],
        "strategy": "parallel",
        "keywords": ["price", "stock"],
        "context_enhanced": {}
    }
    monkeypatch.setattr(home_agent, '_analyze_query', AsyncMock(return_value=mock_analysis_result))

    # Act
    result = await home_agent.process_query(query, {})

    # Assert
    # Check that both agents were called
    mock_specialized_agents["PricingAgent"].process_query.assert_awaited_once()
    mock_specialized_agents["InventoryAgent"].process_query.assert_awaited_once()
    # Check that the customer agent was not
    mock_specialized_agents["CustomerAgent"].process_query.assert_not_awaited()

    assert result["status"] == "success"
    assert "PricingAgent" in result["response"]["detailed_insights"]
    assert "InventoryAgent" in result["response"]["detailed_insights"]

@pytest.mark.asyncio
async def test_process_query_error_handling(home_agent, monkeypatch):
    """
    Tests that if the analysis or handling step fails, the main process_query
    method catches the exception and returns a proper error response.
    """
    # Arrange
    query = "this will fail"
    # Mock the analysis step to raise an exception
    monkeypatch.setattr(home_agent, '_analyze_query', AsyncMock(side_effect=Exception("Simulated analysis failure")))

    # Act
    result = await home_agent.process_query(query, {})
    
    # Assert
    assert result["status"] == "error"
    assert result["error"] == "Simulated analysis failure"
    assert "fallback_response" in result
