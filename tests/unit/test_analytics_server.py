import os
import sys
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mcp_servers.analytics_server import app

MOCK_PRODUCTS = [
    {
        "product_id": "MF001",
        "name": "Classic Wool Trench Coat",
        "brand": "Meridian Fashion",
        "price": 100.00,
        "stock_level": 10,
    },
    {
        "product_id": "ST001",
        "name": "Graphic Print Hoodie",
        "brand": "Stratus",
        "price": 50.00,
        "stock_level": 20,
    },
    {
        "product_id": "ST002",
        "name": "Distressed Denim Jeans",
        "brand": "Stratus",
        "price": 75.00,
        "stock_level": 5,
    },
]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("db.json_store.load_products", lambda: MOCK_PRODUCTS)

    with TestClient(app) as test_client:
        yield test_client


def test_health_check_reports_json_backend(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["data_backend"] == "json"
    response = client.post("/invoke", json={"tool_name": "get_total_inventory_value"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["result"]["total_stock_value_zar"] == 2375.0
    assert data["result"]["total_product_count"] == 3
    assert data["result"]["average_value_per_product"] == pytest.approx(791.67)


def test_invoke_get_product_count_by_brand_success(client):
    payload = {
        "tool_name": "get_product_count_by_brand",
        "input_data": {"brand_name": "Stratus"},
    }
    response = client.post("/invoke", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["result"]["brand"] == "Stratus"
    assert data["result"]["product_count"] == 2


def test_invoke_non_existent_tool_returns_404(client):
    response = client.post(
        "/invoke", json={"tool_name": "calculate_the_meaning_of_life"}
    )

    assert response.status_code == 404
    assert "Tool 'calculate_the_meaning_of_life' not found" in response.json()["detail"]


def test_invoke_tool_with_missing_brand_name_input_returns_400(client):
    payload = {"tool_name": "get_product_count_by_brand", "input_data": {}}
    response = client.post("/invoke", json=payload)

    assert response.status_code == 400
    assert "Missing 'brand_name' for this tool" in response.json()["detail"]


def test_invoke_get_customer_profile_success(client, monkeypatch):
    monkeypatch.setattr(
        "db.service.get_customer_profile",
        AsyncMock(
            return_value={
                "customer_id": "CUST001",
                "first_name": "Sarah",
                "last_name": "Johnson",
            }
        ),
    )
    payload = {
        "tool_name": "get_customer_profile",
        "input_data": {"customer_id": "CUST001"},
    }
    response = client.post("/invoke", json=payload)

    assert response.status_code == 200
    assert response.json()["result"]["first_name"] == "Sarah"


def test_invoke_search_customers_by_name_success(client, monkeypatch):
    monkeypatch.setattr(
        "db.service.search_customers_by_name",
        AsyncMock(return_value=[{"customer_id": "CUST001", "first_name": "Sarah"}]),
    )
    payload = {"tool_name": "search_customers_by_name", "input_data": {"name": "Sarah"}}
    response = client.post("/invoke", json=payload)

    assert response.status_code == 200
    assert response.json()["result"][0]["customer_id"] == "CUST001"
