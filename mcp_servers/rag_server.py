# mcp_servers/rag_server.py

import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

from config.settings import settings
from rag.service import get_rag_service

logger = logging.getLogger(__name__)
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8002"))


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


app = FastAPI(
    title="RAG MCP Server",
    description="Provides standardized access to the RAG system (Milvus).",
    version="1.0.0",
)


def retrieve_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve documents from Milvus with graceful fallback."""
    service = get_rag_service()
    return service.retrieve_documents(query=query, top_k=top_k)


@app.get("/healthz")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "milvus_uri": settings.milvus_uri}


@app.post("/invoke", response_model=ToolOutput)
async def invoke_tool(payload: ToolInput):
    if payload.tool_name == "retrieve_documents":
        query = payload.input_data.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Missing 'query' in input_data.")

        top_k = int(payload.input_data.get("top_k", 5))
        documents = retrieve_documents(query=query, top_k=top_k)
        return ToolOutput(result=documents)

    raise HTTPException(status_code=404, detail=f"Tool '{payload.tool_name}' not found.")


if __name__ == "__main__":
    logger.info(
        "Starting RAG MCP Server on port %s (Milvus: %s)",
        MCP_SERVER_PORT,
        settings.milvus_uri,
    )
    uvicorn.run(app, host="0.0.0.0", port=MCP_SERVER_PORT)
