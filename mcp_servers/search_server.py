# mcp_servers/search_server.py

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List

# --- Configuration ---
# It's crucial to set the TAVILY_API_KEY in your environment
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", 8003)) # Use a unique port

# --- Pydantic Models ---
class ToolInput(BaseModel):
    tool_name: str
    input_data: Dict[str, Any]

class SearchResult(BaseModel):
    title: str
    url: str
    content: str

class ToolOutput(BaseModel):
    status: str = "success"
    result: List[SearchResult]

# --- FastAPI Application ---
app = FastAPI(
    title="Search MCP Server",
    description="Provides standardized access to real-time web search (Tavily).",
    version="1.0.0"
)

# --- Core Logic (Simulated) ---
def _simulate_tavily_search(query: str) -> List[Dict]:
    """
    Simulates making a call to the Tavily Search API.
    A real implementation would use the Tavily Python client.
    """
    if not TAVILY_API_KEY:
        print("Warning: TAVILY_API_KEY is not set. Simulation will proceed without it.")
        api_key_status = "NOT SET"
    else:
        api_key_status = f"SET (ends with ...{TAVILY_API_KEY[-4:]})"

    print(f"Simulating Tavily API search with key status: {api_key_status}")
    print(f"Received search query: '{query}'")

    # Fake search results for demonstration
    return [
        {
            "title": "Latest Winter Fashion Trends in South Africa - Fashion Weekly",
            "url": "https://fake-fashion-weekly.com/trends-sa-winter-2025",
            "content": "This winter in South Africa, expect to see a rise in 'utilitarian chic'. Think long trench coats, chunky boots, and durable fabrics...",
        },
        {
            "title": "Consumer spending habits shift towards experiences - FinReport SA",
            "url": "https://fake-finreport-sa.co.za/consumer-habits-2025",
            "content": "A recent report indicates a significant shift in South African consumer spending, with more disposable income being allocated to experiences rather than physical goods...",
        },
    ]

# --- API Endpoint ---
@app.post("/invoke", response_model=ToolOutput)
async def invoke_tool(payload: ToolInput):
    """
    Invokes the 'web_search' tool.
    """
    if payload.tool_name == "web_search":
        query = payload.input_data.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Missing 'query' in input_data.")
        
        search_results = _simulate_tavily_search(query)
        
        return ToolOutput(result=search_results)
    else:
        raise HTTPException(status_code=404, detail=f"Tool '{payload.tool_name}' not found.")

# --- Main entry point ---
if __name__ == "__main__":
    print(f"Starting Search MCP Server on port {MCP_SERVER_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=MCP_SERVER_PORT)