# mcp_servers/analytics_server.py

import json
import logging
import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config.settings import settings

logger = logging.getLogger(__name__)
DATA_PATH = Path(__file__).resolve().parents[1] / "data"
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8004"))


class ToolInput(BaseModel):
    tool_name: str
    input_data: dict[str, Any] = {}


class ToolOutput(BaseModel):
    status: str = "success"
    result: dict[str, Any]


app = FastAPI(
    title="Analytics MCP Server",
    description="Provides standardized access to business analytics tools.",
    version="1.0.0",
)


def _load_products() -> list[dict[str, Any]]:
    products_file = DATA_PATH / "meridian_products.json"
    with products_file.open(encoding="utf-8") as handle:
        return json.load(handle)


def _get_total_inventory_value() -> dict[str, Any]:
    products = _load_products()
    total_value = sum(p.get("price", 0) * p.get("stock_level", 0) for p in products)
    product_count = len(products)
    return {
        "total_stock_value_zar": round(total_value, 2),
        "total_product_count": product_count,
        "average_value_per_product": round(total_value / product_count if product_count else 0, 2),
    }


def _get_product_count_by_brand(brand_name: str) -> dict[str, Any]:
    products = _load_products()
    count = sum(1 for p in products if p.get("brand", "").lower() == brand_name.lower())
    return {"brand": brand_name, "product_count": count}


def _get_product_details(product_id: str) -> dict[str, Any]:
    for product in _load_products():
        if product.get("product_id") == product_id:
            return product
    return {"product_id": product_id, "error": "Product not found"}


def _get_demand_analytics(product_id: str) -> dict[str, Any]:
    product = _get_product_details(product_id)
    if product.get("error"):
        return product

    stock_level = product.get("stock_level", 0)
    if stock_level >= 100:
        demand = "high"
        trend = "stable"
    elif stock_level >= 40:
        demand = "medium"
        trend = "increasing"
    else:
        demand = "low"
        trend = "decreasing"

    return {
        "product_id": product_id,
        "current_demand": demand,
        "trend": trend,
        "stock_level": stock_level,
        "price": product.get("price"),
        "brand": product.get("brand"),
    }


@app.get("/healthz")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/invoke", response_model=ToolOutput)
async def invoke_tool(payload: ToolInput):
    tool_name = payload.tool_name

    if tool_name == "get_total_inventory_value":
        return ToolOutput(result=_get_total_inventory_value())

    if tool_name == "get_product_count_by_brand":
        brand_name = payload.input_data.get("brand_name")
        if not brand_name:
            raise HTTPException(status_code=400, detail="Missing 'brand_name' for this tool.")
        return ToolOutput(result=_get_product_count_by_brand(brand_name))

    if tool_name == "get_product_details":
        product_id = payload.input_data.get("product_id")
        if not product_id:
            raise HTTPException(status_code=400, detail="Missing 'product_id' for this tool.")
        return ToolOutput(result=_get_product_details(product_id))

    if tool_name == "get_demand_analytics":
        product_id = payload.input_data.get("product_id")
        if not product_id:
            raise HTTPException(status_code=400, detail="Missing 'product_id' for this tool.")
        return ToolOutput(result=_get_demand_analytics(product_id))

    raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found.")


if __name__ == "__main__":
    logger.info("Starting Analytics MCP Server on port %s", MCP_SERVER_PORT)
    uvicorn.run(app, host="0.0.0.0", port=MCP_SERVER_PORT)
