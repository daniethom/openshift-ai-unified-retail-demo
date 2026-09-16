import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mcp_servers.search_server import app


@pytest.fixture
def mock_tavily_search(monkeypatch):
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

    def mock_search(query: str, max_results: int | None = None):
        return mock_results

    monkeypatch.setattr("mcp_servers.search_server.tavily_search", mock_search)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_invoke_web_search_success(client, mock_tavily_search):
    payload = {
        "tool_name": "web_search",
        "input_data": {"query": "latest fashion trends"},
    }
    response = client.post("/invoke", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["result"]) == 2
    assert data["result"][0]["title"] == "Mock Search Result 1"
    assert data["result"][1]["url"] == "https://example.com/result2"


def test_invoke_search_with_missing_query_returns_400(client):
    payload = {"tool_name": "web_search", "input_data": {}}
    response = client.post("/invoke", json=payload)

    assert response.status_code == 400
    assert "Missing 'query' in input_data" in response.json()["detail"]


def test_invoke_non_existent_tool_returns_404(client):
    payload = {"tool_name": "discover_new_planets", "input_data": {}}
    response = client.post("/invoke", json=payload)

    assert response.status_code == 404
    assert "Tool 'discover_new_planets' not found" in response.json()["detail"].rstrip(
        "."
    )
