# mcp_servers/analytics_server.py

import logging
import os
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config.settings import settings
from db import service as data_service

logger = logging.getLogger(__name__)
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8004"))


class ToolInput(BaseModel):
    tool_name: str
    input_data: dict[str, Any] = {}


class ToolOutput(BaseModel):
    status: str = "success"
    result: dict[str, Any] | list[dict[str, Any]]


app = FastAPI(
    title="Analytics MCP Server",
    description="Provides standardized access to business analytics tools.",
    version="1.0.0",
)


@app.get("/healthz")
async def health_check() -> dict[str, str]:
    backend = (
        "json"
        if settings.use_json_fallback or not settings.database_url
        else "postgres"
    )
    return {"status": "ok", "data_backend": backend}


@app.post("/invoke", response_model=ToolOutput)
async def invoke_tool(payload: ToolInput):
    tool_name = payload.tool_name

    if tool_name == "get_total_inventory_value":
        return ToolOutput(result=await data_service.get_total_inventory_value())

    if tool_name == "get_product_count_by_brand":
        brand_name = payload.input_data.get("brand_name")
        if not brand_name:
            raise HTTPException(
                status_code=400, detail="Missing 'brand_name' for this tool."
            )
        return ToolOutput(
            result=await data_service.get_product_count_by_brand(brand_name)
        )

    if tool_name == "get_product_details":
        product_id = payload.input_data.get("product_id")
        if not product_id:
            raise HTTPException(
                status_code=400, detail="Missing 'product_id' for this tool."
            )
        return ToolOutput(result=await data_service.get_product_details(product_id))

    if tool_name == "get_demand_analytics":
        product_id = payload.input_data.get("product_id")
        if not product_id:
            raise HTTPException(
                status_code=400, detail="Missing 'product_id' for this tool."
            )
        return ToolOutput(result=await data_service.get_demand_analytics(product_id))

    if tool_name == "get_customer_profile":
        customer_id = payload.input_data.get("customer_id")
        if not customer_id:
            raise HTTPException(
                status_code=400, detail="Missing 'customer_id' for this tool."
            )
        return ToolOutput(result=await data_service.get_customer_profile(customer_id))

    if tool_name == "search_customers_by_name":
        name = payload.input_data.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="Missing 'name' for this tool.")
        return ToolOutput(result=await data_service.search_customers_by_name(name))

    raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found.")


if __name__ == "__main__":
    logger.info("Starting Analytics MCP Server on port %s", MCP_SERVER_PORT)
    uvicorn.run(app, host="0.0.0.0", port=MCP_SERVER_PORT)
