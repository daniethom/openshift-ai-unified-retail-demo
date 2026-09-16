import os
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mcp_servers.llm_server import app


@pytest.fixture
def mock_openai_client(monkeypatch):
    mock_message = MagicMock()
    mock_message.content = "This is a simulated response from the LLM."
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion
    monkeypatch.setattr("mcp_servers.llm_server.client", mock_client)
    return mock_client


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_invoke_generate_text_success(client, mock_openai_client):
    response = client.post("/invoke", json={"prompt": "Tell me about winter fashion."})

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "This is a simulated response from the LLM."
    mock_openai_client.chat.completions.create.assert_called_once()


def test_invoke_generate_text_with_missing_prompt_returns_422(client, mock_openai_client):
    response = client.post("/invoke", json={})

    assert response.status_code == 422


def test_invoke_llm_without_client_returns_500(client, monkeypatch):
    monkeypatch.setattr("mcp_servers.llm_server.client", None)
    response = client.post("/invoke", json={"prompt": "Hello"})

    assert response.status_code == 500
    assert "LLM client not initialized" in response.json()["detail"]
