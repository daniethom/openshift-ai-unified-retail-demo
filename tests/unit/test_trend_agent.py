import pytest
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# It's important to import the Agent *after* the path has been modified
from agents.trend_agent import TrendAgent

# --- Pytest Fixtures ---

@pytest.fixture
def mock_mcp_servers():
    """Provides a dictionary of mock MCP servers for the TrendAgent."""
    search_server = MagicMock()
    # Mock the async search method
    search_server.search = AsyncMock(return_value={"trends": [{"name": "mock_trend"}]})

    rag_server = MagicMock()
    # Mock the async retrieve_documents method
    rag_server.retrieve_documents = AsyncMock(return_value=[{"source": "mock_report.pdf"}])
    
    return {
        "search_server": search_server,
        "rag_server": rag_server,
    }

@pytest.fixture
def mock_data_store():
    """Provides a mock for the data_store dependency."""
    return MagicMock()

@pytest.fixture
def trend_agent(mock_mcp_servers, mock_data_store):
    """Provides an instance of the TrendAgent with all dependencies mocked."""
    agent = TrendAgent(mcp_servers=mock_mcp_servers, data_store=mock_data_store)
    return agent

# --- Test Cases ---

@pytest.mark.parametrize("query, expected_type", [
    ("what are the winter trends?", "seasonal_analysis"),
    ("analyze emerging micro trends", "micro_macro"),
    ("what is zara doing with utility wear?", "competitor_watch"),
    ("general trend analysis", "general_analysis")
])
def test_classify_trend_query(trend_agent, query, expected_type):
    """
    Tests the query classification logic to ensure it correctly identifies the query type.
    """
    # Act
    query_type = trend_agent._classify_trend_query(query)
    
    # Assert
    assert query_type == expected_type

@pytest.mark.asyncio
async def test_process_query_handles_seasonal_analysis(trend_agent, mock_mcp_servers):
    """
    Tests the end-to-end processing of a seasonal analysis query.
    Verifies that the agent calls its dependencies (the MCP servers) correctly.
    """
    # Arrange
    query = "What are the key trends for summer in Cape Town?"
    context = {"season": "summer", "region": "Cape Town", "category": "womenswear"}
    
    # Act
    result = await trend_agent.process_query(query, context)
    
    # Assert
    assert result["status"] == "success"
    assert result["query_type"] == "seasonal_analysis"
    
    # Verify that the mocked MCP servers were called
    mock_mcp_servers["search_server"].search.assert_awaited_once()
    mock_mcp_servers["rag_server"].retrieve_documents.assert_awaited_once()
    
    # Check the structure of the result
    assert "top_trends" in result["result"]
    assert "recommendations" in result["result"]
    assert result["result"]["top_trends"][0]["name"] == "Power Suiting" # Based on mock synthesis

@pytest.mark.asyncio
async def test_process_query_routes_to_competitor_watch(trend_agent, monkeypatch):
    """
    Tests that the main process_query router correctly calls the
    competitor watch handler when a relevant query is made.
    """
    # Arrange
    query = "What is H&M doing with Gorpcore?"
    context = {}
    
    # Mock the specific handler to confirm it gets called
    mock_handler = AsyncMock(return_value={"competitor_adoption": []})
    monkeypatch.setattr(trend_agent, '_handle_competitor_watch', mock_handler)
    
    # Act
    await trend_agent.process_query(query, context)
    
    # Assert
    mock_handler.assert_awaited_once_with(query, context)

@pytest.mark.asyncio
async def test_process_query_general_error_handling(trend_agent, monkeypatch):
    """
    Tests that if any step in the process fails, the main process_query method
    catches the exception and returns a properly formatted error response.
    """
    # Arrange
    query = "This query is destined to fail."
    
    # Mock the query classification to raise an error
    monkeypatch.setattr(trend_agent, '_classify_trend_query', MagicMock(side_effect=Exception("Classification failed")))
    
    # Act
    result = await trend_agent.process_query(query, {})
    
    # Assert
    assert result["status"] == "error"
    assert "Classification failed" in result["error"]
