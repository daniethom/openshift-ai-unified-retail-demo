# mcp_servers/rag_server.py

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List

# --- Configuration ---
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus-service")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", 19530))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", 8002)) # Use a unique port

# --- Pydantic Models ---
class ToolInput(BaseModel):
    tool_name: str
    input_data: Dict[str, Any]

class DocumentSnippet(BaseModel):
    source: str
    content: str
    score: float

class ToolOutput(BaseModel):
    status: str = "success"
    result: List[DocumentSnippet]

# --- FastAPI Application ---
app = FastAPI(
    title="RAG MCP Server",
    description="Provides standardized access to the RAG system (Milvus).",
    version="1.0.0"
)

# --- Core Logic (Simulated) ---
def _simulate_rag_retrieval(query: str, top_k: int = 3) -> List[Dict]:
    """
    Simulates connecting to Milvus and retrieving documents.
    A real implementation would use the Milvus client library to perform a
    vector search based on the query embeddings.
    """
    print(f"Simulating RAG retrieval from: {MILVUS_HOST}:{MILVUS_PORT}")
    print(f"Using embedding model: {EMBEDDING_MODEL}")
    print(f"Received query: '{query}'")

    # Fake retrieved documents for demonstration
    fake_docs = [
        {"source": "docs/DEMO_GUIDE.md", "content": "The Trend Agent analyzes global and local fashion trends...", "score": 0.91},
        {"source": "data/fashion_trends.json", "content": "Power Tailoring & Suiting: A focus on sharply tailored blazers...", "score": 0.88},
        {"source": "data/sa_market_data.json", "content": "Cape Town's market shows a preference for a laid-back, quality-focused lifestyle...", "score": 0.85},
    ]
    return fake_docs[:top_k]

# --- API Endpoint ---
@app.post("/invoke", response_model=ToolOutput)
async def invoke_tool(payload: ToolInput):
    """
    Invokes the 'retrieve_documents' tool.
    """
    if payload.tool_name == "retrieve_documents":
        query = payload.input_data.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Missing 'query' in input_data.")
        
        documents = _simulate_rag_retrieval(query)
        
        return ToolOutput(result=documents)
    else:
        raise HTTPException(status_code=404, detail=f"Tool '{payload.tool_name}' not found.")

# --- Main entry point ---
if __name__ == "__main__":
    print(f"Starting RAG MCP Server on port {MCP_SERVER_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=MCP_SERVER_PORT)