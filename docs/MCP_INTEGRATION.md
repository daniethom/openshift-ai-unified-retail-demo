# MCP (Model Context Protocol) Integration

## 1. Purpose

MCP decouples AI agents from backend tool implementations. Agents call a consistent HTTP API; MCP servers wrap Milvus, Tavily, kServe, and internal JSON analytics.

Benefits:

- **Modularity** — swap backends without rewriting agents
- **Simplicity** — agents use `agents/mcp_client.py`, not vendor SDKs
- **Scalability** — each MCP server is an independent Deployment on OpenShift
- **Extensibility** — add tools by extending server `/invoke` handlers

## 2. Agent integration

Agents call MCP through **`agents/mcp_client.py`**:

```python
from agents import mcp_client

results = await mcp_client.web_search("winter fashion Cape Town")
docs = await mcp_client.retrieve_documents("power suiting trends", top_k=5)
text = await mcp_client.invoke_llm("Summarize these trends...")
analytics = await mcp_client.get_product_details("MF001")
```

URLs are loaded from **`config/settings.py`**:

- `LLM_MCP_URL`, `RAG_MCP_URL`, `SEARCH_MCP_URL`, `ANALYTICS_MCP_URL`

On OpenShift these resolve to in-cluster Services (e.g. `http://rag-server:8002`).

## 3. Server overview

| File | Health | Backend |
|------|--------|---------|
| `mcp_servers/llm_server.py` | `GET /healthz` | kServe via OpenAI client (`LLM_API_BASE`) |
| `mcp_servers/rag_server.py` | `GET /healthz` | Milvus (`rag/service.py`) |
| `mcp_servers/search_server.py` | `GET /healthz` | Tavily API |
| `mcp_servers/analytics_server.py` | `GET /healthz` | `data/*.json` |

## 4. API contract

### Tool-based servers (RAG, Search, Analytics)

**Endpoint:** `POST /invoke`

**Request:**

```json
{
  "tool_name": "name_of_the_tool",
  "input_data": {
    "parameter1": "value1"
  }
}
```

**Response:**

```json
{
  "status": "success",
  "result": { }
}
```

`result` shape varies by tool (object or array).

### LLM server

**Endpoint:** `POST /invoke`

**Request:**

```json
{
  "prompt": "Your prompt here"
}
```

**Response:**

```json
{
  "response": "Generated text"
}
```

## 5. Server tools

### Analytics server

| Tool | input_data | Description |
|------|------------|-------------|
| `get_total_inventory_value` | `{}` | Sum of stock value across products |
| `get_product_count_by_brand` | `brand_name` | Count products by brand |
| `get_product_details` | `product_id` | Product record from JSON |
| `get_demand_analytics` | `product_id` | Derived demand metrics |

**Example:**

```json
{
  "tool_name": "get_total_inventory_value",
  "input_data": {}
}
```

### Search server

| Tool | input_data | Description |
|------|------------|-------------|
| `web_search` | `query`, optional `max_results` | Tavily web search |

Returns array of `{title, url, content}`. Falls back to placeholder results if `TAVILY_API_KEY` is unset.

**Example:**

```json
{
  "tool_name": "web_search",
  "input_data": {
    "query": "latest fashion trends in south africa"
  }
}
```

### RAG server

| Tool | input_data | Description |
|------|------------|-------------|
| `retrieve_documents` | `query`, optional `top_k` | Milvus vector search |

Returns array of `{source, content, score}`. Falls back to static docs if Milvus unavailable or `RAG_USE_FALLBACK=true`.

**Example:**

```json
{
  "tool_name": "retrieve_documents",
  "input_data": {
    "query": "winter fashion trends professional women",
    "top_k": 5
  }
}
```

### LLM server

Proxies to the OpenAI-compatible endpoint configured in `LLM_API_BASE` / `LLM_MODEL_NAME`.

**Example:**

```json
{
  "prompt": "Summarize Q4 outerwear performance for an executive."
}
```

## 6. Configuration reference

| Variable | Used by | Description |
|----------|---------|-------------|
| `LLM_MCP_URL` | Agents | LLM MCP service URL |
| `LLM_API_BASE` | LLM MCP server | kServe `/v1` endpoint |
| `LLM_API_KEY` | LLM MCP server | API token (often `not-needed` in-cluster) |
| `LLM_MODEL_NAME` | LLM MCP server | Served model name |
| `MILVUS_URI` | RAG service | Milvus connection |
| `MILVUS_COLLECTION_NAME` | RAG service | Collection name |
| `TAVILY_API_KEY` | Search server | Tavily authentication |
| `TAVILY_MAX_RESULTS` | Search server | Max results per query |
| `RAG_USE_FALLBACK` | RAG service | Skip Milvus; use static docs |

## 7. Testing MCP servers locally

```bash
# Terminal 1–4: start servers
uvicorn mcp_servers.llm_server:app --port 8001
uvicorn mcp_servers.rag_server:app --port 8002
uvicorn mcp_servers.search_server:app --port 8003
uvicorn mcp_servers.analytics_server:app --port 8004

# Health check
curl http://127.0.0.1:8002/healthz

# Invoke RAG
curl -X POST http://127.0.0.1:8002/invoke \
  -H "Content-Type: application/json" \
  -d '{"tool_name":"retrieve_documents","input_data":{"query":"winter trends"}}'
```

Unit tests: `tests/unit/test_rag_server.py`, `test_search_server.py`, `test_mcp_client.py`.

## 8. Related documentation

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [DEPLOYMENT.md](DEPLOYMENT.md)
