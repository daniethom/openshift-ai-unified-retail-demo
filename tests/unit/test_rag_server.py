import pytest
from fastapi.testclient import TestClient
import sys
import os
from typing import List, Dict

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the app
from mcp_servers.rag_server import app

# --- Pytest Fixtures ---

@pytest.fixture
def mock_rag_retrieval(monkeypatch):
    """
    A fixture to monkeypatch the actual RAG retrieval function,
    preventing real database calls and providing a predictable, mock response.
    """
    mock_documents: List[Dict] = [
        {"source": "mock_file1.md", "content": "This is a mock document about winter coats.", "score": 0.95},
        {"source": "mock_file2.json", "content": "Another mock document about professional attire.", "score": 0.92},
    ]

    def mock_retrieval(query: str, top_k: int = 3) -> List[Dict]:
        print(f"Mock RAG retrieval with query: {query}")
        return mock_documents

    # Replace the function in the server module with our mock version
    monkeypatch.setattr("mcp_servers.rag_server._simulate_rag_retrieval", mock_retrieval)

@pytest.fixture
def client():
    """Provides a TestClient for our FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client

# --- Test Cases ---

def test_invoke_retrieve_documents_success(client, mock_rag_retrieval):
    """
    Tests a successful call to the 'retrieve_documents' tool.
    Asserts a 200 OK status and that the mock documents are returned.
    """
    # Arrange
    payload = {
        "tool_name": "retrieve_documents",
        "input_data": {"query": "Tell me about winter fashion for professionals."}
    }
    
    # Act
    response = client.post("/invoke", json=payload)
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["result"]) == 2
    assert data["result"][0]["source"] == "mock_file1.md"
    assert data["result"][1]["score"] == 0.92

def test_invoke_retrieve_documents_with_missing_query_returns_400(client):
    """
    Tests that calling 'retrieve_documents' without a 'query' in the input_data
    returns a 400 Bad Request error.
    """
    # Arrange
    payload = {
        "tool_name": "retrieve_documents",
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
        "tool_name": "format_hard_drive"
    }
    
    # Act
    response = client.post("/invoke", json=payload)
    
    # Assert
    assert response.status_code == 404
    assert "Tool 'format_hard_drive' not found" in response.json()["detail"]

