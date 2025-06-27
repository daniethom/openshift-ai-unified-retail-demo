import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the app
from mcp_servers.search_server import app

# --- Pytest Fixtures ---

@pytest.fixture
def mock_tavily_search(monkeypatch):
    """
    A fixture to monkeypatch the actual search function, preventing
    real network calls and providing a predictable, mock response.
    """
    mock_results = [
        {
            "title": "Mock Search Result 1",
            "url": "https://example.com/result1",
            "content": "This is the first mock search result content.",
        },
        {
            "title": "Mock Search Result 2",
            "url": "https://example.com/result2",
            "content": "This is the second mock search result content.",
        },
    ]

    def mock_search(query: str):
        print(f"Mock search called with query: {query}")
        return mock_results

    # Replace the function in the server module with our mock version
    monkeypatch.setattr("mcp_servers.search_server._simulate_tavily_search", mock_search)

@pytest.fixture
def client():
    """Provides a TestClient for our FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client

# --- Test Cases ---

def test_invoke_web_search_success(client, mock_tavily_search):
    """
    Tests a successful call to the 'web_search' tool.
    Asserts a 200 OK status and that the mock results are returned.
    """
    # Arrange
    payload = {
        "tool_name": "web_search",
        "input_data": {"query": "latest fashion trends"}
    }
    
    # Act
    response = client.post("/invoke", json=payload)
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["result"]) == 2
    assert data["result"][0]["title"] == "Mock Search Result 1"
    assert data["result"][1]["url"] == "https://example.com/result2"

def test_invoke_search_with_missing_query_returns_400(client):
    """
    Tests that calling 'web_search' without a 'query' in the input_data
    returns a 400 Bad Request error.
    """
    # Arrange
    payload = {
        "tool_name": "web_search",
        "input_data": {} # Missing 'query'
    }
    
    # Act
    response = client.post("/invoke", json=payload)
    
    # Assert
    assert response.status_code == 400
    assert "Missing 'query' in input_data" in response.json()["detail"]

def test_invoke_non_existent_tool_returns_404(client):
    """
    Tests that calling a non-existent tool correctly returns a 404 Not Found error.
    """
    # Arrange
    payload = {
        "tool_name": "discover_new_planets"
    }
    
    # Act
    response = client.post("/invoke", json=payload)
    
    # Assert
    assert response.status_code == 404
    assert "Tool 'discover_new_planets' not found" in response.json()["detail"]