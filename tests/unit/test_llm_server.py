import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the app
from mcp_servers.llm_server import app

# --- Pytest Fixtures ---

@pytest.fixture
def mock_llm_call(monkeypatch):
    """
    A fixture to monkeypatch the actual LLM call function,
    providing a predictable, mock response.
    """
    mock_response_text = "This is a simulated response from the LLM."

    def mock_call(prompt: str):
        print(f"Mock LLM call with prompt: {prompt}")
        return mock_response_text

    # Replace the function in the server module with our mock version
    monkeypatch.setattr("mcp_servers.llm_server._simulate_llm_call", mock_call)

@pytest.fixture
def client():
    """Provides a TestClient for our FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client

# --- Test Cases ---

def test_invoke_generate_text_success(client, mock_llm_call):
    """
    Tests a successful call to the 'generate_text' tool.
    Asserts a 200 OK status and that the mock response is returned.
    """
    # Arrange
    payload = {
        "tool_name": "generate_text",
        "input_data": {"prompt": "Tell me about winter fashion."}
    }
    
    # Act
    response = client.post("/invoke", json=payload)
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["result"]["text"] == "This is a simulated response from the LLM."

def test_invoke_generate_text_with_missing_prompt_returns_400(client):
    """
    Tests that calling 'generate_text' without a 'prompt' in the input_data
    returns a 400 Bad Request error.
    """
    # Arrange
    payload = {
        "tool_name": "generate_text",
        "input_data": {} # Missing 'prompt'
    }
    
    # Act
    response = client.post("/invoke", json=payload)
    
    # Assert
    assert response.status_code == 400
    assert "Missing 'prompt' in input_data" in response.json()["detail"]

def test_invoke_non_existent_tool_returns_404(client):
    """
    Tests that calling a non-existent tool correctly returns a 404 Not Found error.
    """
    # Arrange
    payload = {
        "tool_name": "generate_images"
    }
    
    # Act
    response = client.post("/invoke", json=payload)
    
    # Assert
    assert response.status_code == 404
    assert "Tool 'generate_images' not found" in response.json()["detail"]

