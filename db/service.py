"""Unified data access with PostgreSQL primary and JSON fallback."""

from __future__ import annotations

from typing import Any

from config.settings import settings
from db import json_store
from db.repositories import customers as customer_repo
from db.repositories import insights as insight_repo
from db.repositories import products as product_repo
from db.repositories import trends as trend_repo
from db.session import get_session_factory, session_scope


def _use_json_fallback() -> bool:
    return settings.use_json_fallback or not settings.database_url or get_session_factory() is None


async def get_total_inventory_value() -> dict[str, Any]:
    if _use_json_fallback():
        return json_store.get_total_inventory_value()

    async with session_scope() as session:
        return await product_repo.total_inventory_value(session)


async def get_product_count_by_brand(brand_name: str) -> dict[str, Any]:
    if _use_json_fallback():
        return json_store.get_product_count_by_brand(brand_name)

    async with session_scope() as session:
        count = await product_repo.count_by_brand(session, brand_name)
        return {"brand": brand_name, "product_count": count}


async def get_product_details(product_id: str) -> dict[str, Any]:
    if _use_json_fallback():
        return json_store.get_product_details(product_id)

    async with session_scope() as session:
        product = await product_repo.get_by_id(session, product_id)
        if product is None:
            return {"product_id": product_id, "error": "Product not found"}
        return product_repo.product_to_dict(product)


async def get_demand_analytics(product_id: str) -> dict[str, Any]:
    if _use_json_fallback():
        return json_store.get_demand_analytics(product_id)

    product = await get_product_details(product_id)
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


async def get_customer_profile(customer_id: str) -> dict[str, Any]:
    if _use_json_fallback():
        profile = json_store.get_customer_profile(customer_id)
        if profile is None:
            return {"customer_id": customer_id, "error": "Customer not found"}
        return profile

    async with session_scope() as session:
        customer = await customer_repo.get_by_id(session, customer_id)
        if customer is None:
            return {"customer_id": customer_id, "error": "Customer not found"}
        return customer_repo.customer_to_dict(customer)


async def search_customers_by_name(name: str) -> list[dict[str, Any]]:
    if _use_json_fallback():
        return json_store.search_customers_by_name(name)

    async with session_scope() as session:
        customers = await customer_repo.search_by_name(session, name)
        return [customer_repo.customer_to_dict(customer) for customer in customers]


async def build_knowledge_documents() -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []

    if _use_json_fallback():
        products = json_store.load_products()
        trends = json_store.load_trends()
        insights = json_store.load_insights()
    else:
        async with session_scope() as session:
            products = [
                product_repo.product_to_dict(product)
                for product in await product_repo.get_all(session)
            ]
            trends = [
                trend_repo.trend_to_dict(trend)
                for trend in await trend_repo.get_all(session)
            ]
            insights = [
                insight_repo.insight_to_dict(insight)
                for insight in await insight_repo.get_all(session)
            ]

    for product in products:
        documents.append(
            {
                "id": product["product_id"],
                "content": (
                    f"Product {product['name']} ({product['product_id']}) "
                    f"by {product['brand']}: {product.get('description', '')}"
                ),
                "source": "products",
                "metadata": product,
            }
        )

    for trend in trends:
        documents.append(
            {
                "id": trend["trend_id"],
                "content": f"{trend['title']}: {trend.get('description', '')}",
                "source": "fashion_trends",
                "metadata": trend,
            }
        )

    for insight in insights:
        documents.append(
            {
                "id": insight["insight_id"],
                "content": f"{insight['title']}: {insight.get('summary', '')}",
                "source": "market_insights",
                "metadata": insight,
            }
        )

    return documents
