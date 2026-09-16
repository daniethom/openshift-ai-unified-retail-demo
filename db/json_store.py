"""JSON file access used for seed input and offline fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.settings import settings


def data_dir() -> Path:
    return Path(settings.data_path)


def load_json(filename: str) -> list[dict[str, Any]]:
    path = data_dir() / filename
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else []


def load_products() -> list[dict[str, Any]]:
    return load_json("meridian_products.json")


def load_customers() -> list[dict[str, Any]]:
    return load_json("customers.json")


def load_trends() -> list[dict[str, Any]]:
    return load_json("fashion_trends.json")


def load_insights() -> list[dict[str, Any]]:
    return load_json("sa_market_data.json")


def get_total_inventory_value() -> dict[str, Any]:
    products = load_products()
    total_value = sum(p.get("price", 0) * p.get("stock_level", 0) for p in products)
    product_count = len(products)
    return {
        "total_stock_value_zar": round(total_value, 2),
        "total_product_count": product_count,
        "average_value_per_product": round(
            total_value / product_count if product_count else 0, 2
        ),
    }


def get_product_count_by_brand(brand_name: str) -> dict[str, Any]:
    products = load_products()
    count = sum(1 for p in products if p.get("brand", "").lower() == brand_name.lower())
    return {"brand": brand_name, "product_count": count}


def get_product_details(product_id: str) -> dict[str, Any]:
    for product in load_products():
        if product.get("product_id") == product_id:
            return product
    return {"product_id": product_id, "error": "Product not found"}


def get_demand_analytics(product_id: str) -> dict[str, Any]:
    product = get_product_details(product_id)
    if product.get("error"):
        return product

    stock_level = product.get("stock_level", 0)
    if stock_level >= 100:
        demand, trend = "high", "stable"
    elif stock_level >= 40:
        demand, trend = "medium", "increasing"
    else:
        demand, trend = "low", "decreasing"

    return {
        "product_id": product_id,
        "current_demand": demand,
        "trend": trend,
        "stock_level": stock_level,
        "price": product.get("price"),
        "brand": product.get("brand"),
    }


def get_customer_profile(customer_id: str) -> dict[str, Any] | None:
    for customer in load_customers():
        if customer.get("customer_id") == customer_id:
            return customer
    return None


def search_customers_by_name(name: str) -> list[dict[str, Any]]:
    needle = name.lower()
    return [
        customer
        for customer in load_customers()
        if needle in customer.get("first_name", "").lower()
        or needle in customer.get("last_name", "").lower()
        or needle
        in f"{customer.get('first_name', '')} {customer.get('last_name', '')}".lower()
    ]
