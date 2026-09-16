"""Tests for JSON fallback data store."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from db import json_store


def test_get_product_details_found():
    product = json_store.get_product_details("MF001")
    assert product["product_id"] == "MF001"
    assert product["brand"] == "Meridian Fashion"


def test_get_product_details_missing():
    product = json_store.get_product_details("UNKNOWN")
    assert product["error"] == "Product not found"


def test_search_customers_by_name():
    matches = json_store.search_customers_by_name("Sarah")
    assert any(match["first_name"] == "Sarah" for match in matches)


def test_get_total_inventory_value_has_counts():
    summary = json_store.get_total_inventory_value()
    assert summary["total_product_count"] > 0
    assert summary["total_stock_value_zar"] > 0


def test_get_demand_analytics_levels():
    high = json_store.get_demand_analytics("MF002")
    assert high["current_demand"] == "high"

    missing = json_store.get_demand_analytics("UNKNOWN")
    assert missing["error"] == "Product not found"
