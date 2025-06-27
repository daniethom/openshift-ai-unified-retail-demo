"""
LLM MCP Server - Model Context Protocol server for LLM integration
Provides standardized access to language models on OpenShift AI
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app for MCP server
app = FastAPI(
    title="LLM MCP Server",
    description="Model Context Protocol server for LLM integration",
    version="1.0.0"
)

# Configuration
GRANITE_ENDPOINT = os.getenv("GRANITE_ENDPOINT", "http://granite-service:8080")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "granite-3b")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))


class LLMRequest(BaseModel):
    """Request model for LLM generation"""
    prompt: str = Field(..., description="Input prompt for generation")
    max_tokens: Optional[int] = Field(MAX_TOKENS, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(TEMPERATURE, description="Sampling temperature")
    system_prompt: Optional[str] = Field(None, description="System prompt for context")
    stream: Optional[bool] = Field(False, description="Enable streaming responses")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Response model for LLM generation"""
    text: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model used for generation")
    tokens_used: int = Field(..., description="Number of tokens used")
    generation_time: float = Field(..., description="Time taken for generation")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CompletionRequest(BaseModel):
    """Request for text completion"""
    text: str
    context: Optional[str] = None
    max_length: Optional[int] = 100


class EmbeddingRequest(BaseModel):
    """Request for text embeddings"""
    texts: List[str]
    model: Optional[str] = "all-MiniLM-L6-v2"


class MCPServer:
    """MCP Server for LLM operations"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.model_cache = {}
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_generation_time": 0,
            "total_tokens": 0
        }
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text using LLM"""
        start_time = datetime.now()
        self.metrics["total_requests"] += 1
        
        try:
            # Prepare the request for the model
            if MODEL_NAME.startswith("granite"):
                response = await self._call_granite(request)
            elif OPENAI_API_KEY:
                response = await self._call_openai(request)
            else:
                response = await self._call_local_llm(request)
            
            # Calculate metrics
            generation_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(generation_time, response["tokens_used"])
            
            self.metrics["successful_requests"] += 1
            
            return LLMResponse(
                text=response["text"],
                model=response["model"],
                tokens_used=response["tokens_used"],
                generation_time=generation_time,
                metadata={
                    "request_id": response.get("request_id"),
                    "finish_reason": response.get("finish_reason", "complete")
                }
            )
            
        except Exception as e:
            self.metrics["failed_requests"] += 1
            logger.error(f"LLM generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _call_granite(self, request: LLMRequest) -> Dict[str, Any]:
        """Call Granite model on OpenShift AI"""
        payload = {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream
        }
        
        if request.system_prompt:
            payload["system_prompt"] = request.system_prompt
        
        response = await self.client.post(
            f"{GRANITE_ENDPOINT}/v1/completions",
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Granite API error: {response.text}")
        
        data = response.json()
        return {
            "text": data["choices"][0]["text"],
            "model": "granite-3b",
            "tokens_used": data["usage"]["total_tokens"],
            "request_id": data.get("id"),
            "finish_reason": data["choices"][0].get("finish_reason")
        }
    
    async def _call_openai(self, request: LLMRequest) -> Dict[str, Any]:
        """Call OpenAI API as fallback"""
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream
        }
        
        response = await self.client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.text}")
        
        data = response.json()
        return {
            "text": data["choices"][0]["message"]["content"],
            "model": data["model"],
            "tokens_used": data["usage"]["total_tokens"],
            "request_id": data["id"],
            "finish_reason": data["choices"][0]["finish_reason"]
        }
    
    async def _call_local_llm(self, request: LLMRequest) -> Dict[str, Any]:
        """Call local LLM (mock for demo)"""
        # Mock response for demo when no real LLM is available
        await asyncio.sleep(0.5)  # Simulate processing time
        
        mock_response = f"Based on your request about '{request.prompt[:50]}...', "
        mock_response += "I would recommend focusing on seasonal trends and customer preferences. "
        mock_response += "Our analysis shows strong demand in the current market."
        
        return {
            "text": mock_response,
            "model": "mock-llm",
            "tokens_used": len(mock_response.split()),
            "request_id": "mock-" + str(int(datetime.now().timestamp())),
            "finish_reason": "complete"
        }
    
    def _update_metrics(self, generation_time: float, tokens_used: int):
        """Update server metrics"""
        total = self.metrics["successful_requests"]
        if total > 0:
            current_avg = self.metrics["avg_generation_time"]
            self.metrics["avg_generation_time"] = (
                (current_avg * (total - 1) + generation_time) / total
            )
        else:
            self.metrics["avg_generation_time"] = generation_time
        
        self.metrics["total_tokens"] += tokens_used
    
    async def complete(self, request: CompletionRequest) -> Dict[str, Any]:
        """Complete text based on context"""
        prompt = request.text
        if request.context:
            prompt = f"Context: {request.context}\n\nComplete: {request.text}"
        
        llm_request = LLMRequest(
            prompt=prompt,
            max_tokens=request.max_length,
            temperature=0.7
        )
        
        response = await self.generate(llm_request)
        return {
            "completion": response.text,
            "original": request.text,
            "model": response.model
        }
    
    async def get_embeddings(self, request: EmbeddingRequest) -> Dict[str, Any]:
        """Generate embeddings for texts"""
        # In production, this would call an embedding model
        # For demo, return mock embeddings
        embeddings = []
        for text in request.texts:
            # Simple mock embedding based on text length and content
            embedding = [0.1] * 384  # Standard embedding size
            for i, char in enumerate(text[:384]):
                embedding[i] = ord(char) / 255.0
            embeddings.append(embedding)
        
        return {
            "embeddings": embeddings,
            "model": request.model,
            "dimension": 384
        }
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        prompt = f"Analyze the sentiment of the following text and respond with only one word - positive, negative, or neutral:\n\n{text}"
        
        request = LLMRequest(
            prompt=prompt,
            max_tokens=10,
            temperature=0.1
        )
        
        response = await self.generate(request)
        sentiment = response.text.strip().lower()
        
        # Ensure valid sentiment
        if sentiment not in ["positive", "negative", "neutral"]:
            sentiment = "neutral"
        
        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": 0.85 if sentiment != "neutral" else 0.7
        }
    
    async def summarize(self, text: str, max_length: int = 100) -> Dict[str, Any]:
        """Summarize text"""
        prompt = f"Summarize the following text in no more than {max_length} words:\n\n{text}"
        
        request = LLMRequest(
            prompt=prompt,
            max_tokens=max_length * 2,  # Approximate token count
            temperature=0.5,
            system_prompt="You are a helpful assistant that creates concise, accurate summaries."
        )
        
        response = await self.generate(request)
        return {
            "original_text": text,
            "summary": response.text,
            "original_length": len(text.split()),
            "summary_length": len(response.text.split())
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics"""
        return {
            **self.metrics,
            "uptime": "running",
            "model": MODEL_NAME,
            "endpoint": GRANITE_ENDPOINT if MODEL_NAME.startswith("granite") else "local"
        }


# Initialize MCP server
mcp_server = MCPServer()


# FastAPI endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "LLM MCP Server",
        "status": "running",
        "version": "1.0.0",
        "endpoints": [
            "/generate",
            "/complete",
            "/embeddings",
            "/sentiment",
            "/summarize",
            "/metrics",
            "/health"
        ]
    }


@app.post("/generate", response_model=LLMResponse)
async def generate(request: LLMRequest):
    """Generate text using LLM"""
    return await mcp_server.generate(request)


@app.post("/complete")
async def complete(request: CompletionRequest):
    """Complete text based on context"""
    return await mcp_server.complete(request)


@app.post("/embeddings")
async def embeddings(request: EmbeddingRequest):
    """Generate embeddings for texts"""
    return await mcp_server.get_embeddings(request)


@app.post("/sentiment")
async def sentiment(text: str):
    """Analyze sentiment of text"""
    return await mcp_server.analyze_sentiment(text)


@app.post("/summarize")
async def summarize(text: str, max_length: int = 100):
    """Summarize text"""
    return await mcp_server.summarize(text, max_length)


@app.get("/metrics")
async def metrics():
    """Get server metrics"""
    return mcp_server.get_metrics()


@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Simple health check - ensure we can reach the LLM endpoint
        if MODEL_NAME.startswith("granite"):
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{GRANITE_ENDPOINT}/health")
                llm_healthy = response.status_code == 200
        else:
            llm_healthy = True  # Mock/local LLM always healthy
        
        return {
            "status": "healthy" if llm_healthy else "degraded",
            "llm_endpoint": "reachable" if llm_healthy else "unreachable",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# MCP Protocol Handler
class MCPProtocolHandler:
    """Handles MCP protocol-specific operations"""
    
    @staticmethod
    async def handle_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls"""
        if tool_name == "generate_text":
            request = LLMRequest(**arguments)
            response = await mcp_server.generate(request)
            return response.dict()
        
        elif tool_name == "complete_text":
            request = CompletionRequest(**arguments)
            return await mcp_server.complete(request)
        
        elif tool_name == "get_embeddings":
            request = EmbeddingRequest(**arguments)
            return await mcp_server.get_embeddings(request)
        
        elif tool_name == "analyze_sentiment":
            return await mcp_server.analyze_sentiment(arguments["text"])
        
        elif tool_name == "summarize_text":
            return await mcp_server.summarize(
                arguments["text"],
                arguments.get("max_length", 100)
            )
        
        else:
            raise ValueError(f"Unknown tool: {tool_name}")


# Tool definitions for MCP
MCP_TOOLS = [
    {
        "name": "generate_text",
        "description": "Generate text using LLM",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Input prompt"},
                "max_tokens": {"type": "integer", "description": "Max tokens to generate"},
                "temperature": {"type": "number", "description": "Sampling temperature"},
                "system_prompt": {"type": "string", "description": "System prompt"}
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "complete_text",
        "description": "Complete text based on context",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to complete"},
                "context": {"type": "string", "description": "Context for completion"},
                "max_length": {"type": "integer", "description": "Max completion length"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "get_embeddings",
        "description": "Generate embeddings for texts",
        "parameters": {
            "type": "object",
            "properties": {
                "texts": {"type": "array", "items": {"type": "string"}},
                "model": {"type": "string", "description": "Embedding model to use"}
            },
            "required": ["texts"]
        }
    },
    {
        "name": "analyze_sentiment",
        "description": "Analyze sentiment of text",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "summarize_text",
        "description": "Summarize text",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to summarize"},
                "max_length": {"type": "integer", "description": "Max summary length"}
            },
            "required": ["text"]
        }
    }
]


@app.get("/mcp/tools")
async def get_tools():
    """Get available MCP tools"""
    return {"tools": MCP_TOOLS}


@app.post("/mcp/execute")
async def execute_tool(tool_name: str, arguments: Dict[str, Any]):
    """Execute MCP tool"""
    try:
        result = await MCPProtocolHandler.handle_tool_call(tool_name, arguments)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MCP_SERVER_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)