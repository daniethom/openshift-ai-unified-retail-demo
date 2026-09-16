# Multi-Agent System Design

## 1. Orchestration strategy

The **HomeAgent** (`agents/home_agent.py`) is the central orchestrator. It receives user queries, classifies complexity and intent, delegates to specialist agents, and synthesizes responses.

Specialist agents extend **`agents/base_agent.py`** (CrewAI) and invoke tools through **`agents/mcp_client.py`** — not direct backend SDK calls.

## 2. Collaboration patterns

### Simple query (direct routing)

**Trigger:** Single-domain question.

**Example:** "What is the stock level for product MF001?"

**Execution:** HomeAgent routes to InventoryAgent → analytics/RAG as needed → response to user.

### Parallel execution

**Trigger:** Multiple independent domains.

**Example:** "What is the price and stock level for our winter coats?"

**Execution:** HomeAgent tasks PricingAgent and InventoryAgent concurrently (`asyncio.gather`), then synthesizes.

### Hierarchical delegation

**Trigger:** Multi-step queries where one agent's output feeds the next.

**Example:** "Plan our marketing campaign for the top winter fashion trend."

**Execution:** TrendAgent identifies trends → HomeAgent tasks CustomerAgent with campaign planning using that context.

## 3. Agent roster

| Agent | Role | Key MCP tools |
|-------|------|----------------|
| **HomeAgent** | Chief orchestrator | Coordinates all agents |
| **InventoryAgent** | Stock and supply chain | Analytics (`get_product_details`) |
| **PricingAgent** | Revenue optimization | Search, Analytics (`get_demand_analytics`) |
| **CustomerAgent** | Service and loyalty | RAG, Analytics |
| **TrendAgent** | Fashion intelligence | Search (`web_search`), RAG (`retrieve_documents`) |

### Tool wrappers

- **`agents/tools/search_tools.py`** — `TrendSearcher` → `mcp_client.web_search`
- **`agents/tools/rag_tools.py`** — `MCPRagRetriever` → `mcp_client.retrieve_documents`

## 4. MCP client layer

All HTTP communication with MCP servers flows through **`agents/mcp_client.py`**:

| Function | MCP server | Tool / endpoint |
|----------|------------|-----------------|
| `invoke_llm(prompt)` | LLM | `POST /invoke` with `prompt` |
| `web_search(query)` | Search | `web_search` |
| `retrieve_documents(query, top_k)` | RAG | `retrieve_documents` |
| `get_product_details(id)` | Analytics | `get_product_details` |
| `get_demand_analytics(id)` | Analytics | `get_demand_analytics` |
| `check_mcp_health(url)` | Any | `GET /healthz` |

Configuration URLs come from **`config/settings.py`** (`settings.mcp_servers_config()` for status display).

## 5. Streamlit integration

`streamlit_app/app.py`:

1. Loads settings from environment
2. Initializes specialist agents with MCP endpoint map
3. Creates HomeAgent with agent registry
4. Routes user chat through `HomeAgent.process_query()`

On OpenShift, Streamlit receives MCP URLs from ConfigMaps (`streamlit-ui-config`, `mcp-servers-config`).

## 6. Configuration

Agent behavior is influenced by environment variables (see `.env.example` and `config/settings.py`). No hardcoded `localhost` URLs in production code paths.

Legacy `agents/crew_config.yaml` may contain example endpoints; **`config/settings.py` is the source of truth** for runtime wiring.

## 7. Related documentation

- [MCP_INTEGRATION.md](MCP_INTEGRATION.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [DEMO_GUIDE.md](DEMO_GUIDE.md)
