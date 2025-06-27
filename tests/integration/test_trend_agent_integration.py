import pytest
import uvicorn
import threading
import time
import requests
import sys
import os
from unittest.mock import MagicMock, AsyncMock

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the agent and the server app
from agents.trend_agent import TrendAgent
from mcp_servers.search_server import app as search_app

# --- Pytest Fixtures ---

class UvicornTestServer(uvicorn.Server):
    """A custom Uvicorn server class to run in a background thread."""
    def install_signal_handlers(self):
        # Disable default signal handlers to allow clean shutdown
        pass

    def run_in_thread(self):
        """Runs the server in a separate thread."""
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        # Wait a moment for the server to start up
        while not self.started:
            time.sleep(1e-3)

    def stop(self):
        """Stops the server gracefully."""
        self.should_exit = True
        self.thread.join()

@pytest.fixture(scope="module")
def live_search_server():
    """
    A fixture that starts the Search MCP Server in a background thread,
    yields its URL, and ensures it's shut down after tests are complete.
    """
    host = "127.0.0.1"
    port = 8003 # Use the correct port for the search server
    
    config = uvicorn.Config(search_app, host=host, port=port, log_level="info")
    server = UvicornTestServer(config=config)
    
    # Run the server in a background thread
    server.run_in_thread()
    
    # Yield the base URL to the tests
    yield f"http://{host}:{port}"
    
    # Teardown: stop the server after the tests in the module have run
    server.stop()

@pytest.fixture
def trend_agent_for_integration(live_search_server):
    """
    Provides a TrendAgent instance configured for integration testing.
    - It uses the live search server URL.
    - It mocks other dependencies like the RAG server.
    """
    # Mock the RAG server since we are not testing that integration here
    mock_rag_server = MagicMock()
    mock_rag_server.retrieve_documents = AsyncMock(return_value=[])

    # Create a real HTTP client for the live search server
    # Note: A more advanced setup might use a library like httpx, but for this
    # simple agent, we can mock the direct 'search' method it expects.
    # We will configure the agent's internal tool to point to the live URL.
    
    # We create a mock search server object, but its 'search' method will
    # make a real HTTP request to the live server.
    real_http_search_client = MagicMock()
    
    async def make_real_request(query, **kwargs):
        # This function will be called by the agent's tool.
        # It makes a real HTTP request to the live server.
        payload = {"tool_name": "web_search", "input_data": {"query": query}}
        response = requests.post(f"{live_search_server}/invoke", json=payload)
        response.raise_for_status()
        # The agent's tool expects a dictionary, not a response object.
        return response.json().get("result", {})

    real_http_search_client.search = AsyncMock(side_effect=make_real_request)

    mock_mcp_servers = {
        "search_server": real_http_search_client,
        "rag_server": mock_rag_server, # Still mocked
    }
    
    # Instantiate the agent with the live search client and mocked RAG client
    agent = TrendAgent(mcp_servers=mock_mcp_servers, data_store=MagicMock())
    return agent

# --- Integration Test Case ---

@pytest.mark.asyncio
async def test_trend_agent_integrates_with_search_server(trend_agent_for_integration):
    """
    An integration test to verify the TrendAgent can successfully query
    the live Search MCP Server and process its response.
    """
    # Arrange
    query = "Analyze summer trends for Cape Town"
    context = {"season": "summer", "region": "Cape Town", "category": "womenswear"}
    
    # Act: Call the agent's method that triggers the search
    result = await trend_agent_for_integration._handle_seasonal_analysis(query, context)
    
    # Assert
    # The agent's synthesis logic will process the raw response.
    # We need to check if the data from the search server's simulation made it through.
    # The search_server.py's _simulate_tavily_search returns specific titles.
    assert result is not None
    assert "top_trends" in result
    
    # The agent's _synthesize_trends method will turn the raw search result
    # into a structured trend. We check if a trend derived from the live server's
    # mock response is present.
    # The server's mock response includes "utilitarian chic". Let's check for "Utility & Gorpcore".
    trend_names = [t["name"] for t in result["top_trends"]]
    assert "Utility & Gorpcore" in trend_names

