# mcp_servers/llm_server.py

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any

# --- Configuration ---
# Load configuration from environment variables with sane defaults
GRANITE_ENDPOINT = os.getenv("GRANITE_ENDPOINT", "http://granite-service:8080")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", 8001)) # Use a unique port for this server

# --- Pydantic Models for API Data Structure ---
class ToolInput(BaseModel):
    tool_name: str = Field(..., description="The name of the tool to invoke.")
    input_data: Dict[str, Any] = Field(..., description="The data to be passed to the tool.")

class ToolOutput(BaseModel):
    status: str = "success"
    result: Dict[str, Any]

# --- FastAPI Application ---
app = FastAPI(
    title="LLM MCP Server",
    description="Provides standardized access to Large Language Models.",
    version="1.0.0"
)

# --- Core Logic (Simulated) ---
def _simulate_llm_call(prompt: str) -> str:
    """
    Simulates making a call to the LLM endpoint.
    In a real implementation, this would involve using a library like `requests`
    or `httpx` to send the prompt to the GRANITE_ENDPOINT.
    """
    print(f"Simulating LLM call to: {GRANITE_ENDPOINT}")
    print(f"Received prompt: '{prompt[:50]}...'")
    
    # Fake response for demonstration
    response_text = (
        f"Based on your prompt regarding '{prompt[:20]}...', the analysis suggests focusing on "
        f"sustainable materials for the upcoming summer season, as consumer interest in "
        f"eco-friendly products is at an all-time high."
    )
    return response_text

# --- API Endpoint ---
@app.post("/invoke", response_model=ToolOutput)
async def invoke_tool(payload: ToolInput):
    """
    Invokes a specified tool with the given input data.
    This MCP server currently supports the 'generate_text' tool.
    """
    tool_name = payload.tool_name
    input_data = payload.input_data

    if tool_name == "generate_text":
        prompt = input_data.get("prompt")
        if not prompt:
            raise HTTPException(status_code=400, detail="Missing 'prompt' in input_data.")
        
        # Call the simulated logic
        generated_text = _simulate_llm_call(prompt)
        
        return ToolOutput(result={"text": generated_text})
    else:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found.")

# --- Main entry point for running the server ---
if __name__ == "__main__":
    print(f"Starting LLM MCP Server on port {MCP_SERVER_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=MCP_SERVER_PORT)