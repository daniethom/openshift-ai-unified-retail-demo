# mcp_servers/analytics_server.py

import os
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any

# --- Configuration ---
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", 8004)) # Use a unique port

# --- Pydantic Models ---
class ToolInput(BaseModel):
    tool_name: str
    input_data: Dict[str, Any] = {}

class ToolOutput(BaseModel):
    status: str = "success"
    result: Dict[str, Any]

# --- FastAPI Application ---
app = FastAPI(
    title="Analytics MCP Server",
    description="Provides standardized access to business analytics tools.",
    version="1.0.0"
)

# --- Core Logic (Simulated) ---
def _get_total_inventory_value() -> Dict[str, Any]:
    """
    Simulates loading products.json and calculating total stock value.
    """
    products_file = os.path.join(DATA_PATH, 'meridian_products.json')
    print(f"Loading data from: {products_file}")
    
    try:
        with open(products_file, 'r') as f:
            products = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="meridian_products.json not found.")
    
    total_value = sum(p.get('price', 0) * p.get('stock_level', 0) for p in products)
    product_count = len(products)
    
    return {
        "total_stock_value_zar": round(total_value, 2),
        "total_product_count": product_count,
        "average_value_per_product": round(total_value / product_count if product_count else 0, 2)
    }

def _get_product_count_by_brand(brand_name: str) -> Dict[str, Any]:
    """
    Simulates loading products.json and counting products for a given brand.
    """
    products_file = os.path.join(DATA_PATH, 'meridian_products.json')
    print(f"Loading data from: {products_file} for brand: {brand_name}")
    
    try:
        with open(products_file, 'r') as f:
            products = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="meridian_products.json not found.")
        
    count = sum(1 for p in products if p.get('brand', '').lower() == brand_name.lower())
    
    return {"brand": brand_name, "product_count": count}


# --- API Endpoint ---
@app.post("/invoke", response_model=ToolOutput)
async def invoke_tool(payload: ToolInput):
    """

    Invokes an analytics tool. Supported tools:
    - 'get_total_inventory_value'
    - 'get_product_count_by_brand' (requires 'brand_name' in input_data)
    """
    tool_name = payload.tool_name
    
    if tool_name == "get_total_inventory_value":
        result = _get_total_inventory_value()
        return ToolOutput(result=result)
        
    elif tool_name == "get_product_count_by_brand":
        brand_name = payload.input_data.get("brand_name")
        if not brand_name:
            raise HTTPException(status_code=400, detail="Missing 'brand_name' for this tool.")
        result = _get_product_count_by_brand(brand_name)
        return ToolOutput(result=result)
        
    else:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found.")

# --- Main entry point ---
if __name__ == "__main__":
    print(f"Starting Analytics MCP Server on port {MCP_SERVER_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=MCP_SERVER_PORT)