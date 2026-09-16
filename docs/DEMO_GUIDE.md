# Meridian Retail AI — Live Demo Guide

## 1. Introduction script

**(Presenter):** "Welcome, everyone. Today we're demonstrating the Meridian Retail AI system — a multi-agent platform running on Red Hat OpenShift AI.

This isn't just a chatbot. It's a collaborative crew of specialized AI agents — experts in pricing, inventory, customer service, and trends — working together to solve complex retail problems. We'll show how the system moves beyond simple answers to deliver cross-functional, actionable insights."

---

## 2. Demo-day checklist

Run this **before** the audience arrives:

```bash
# Logged into OpenShift
oc whoami

# Full readiness report (routes, pods, sample queries)
make demo-checklist

# Strict mode — fails if anything is wrong
make demo-checklist-strict
```

The checklist prints:

- Pre-demo manual items (secrets, model download, Milvus priming)
- Pod and deployment status
- Streamlit HTTPS URL
- kServe / Granite status (production)
- Background job status
- All four sample queries below
- Useful `oc logs` commands

**Production first-time setup:**

```bash
make deploy-prod-full
make demo-checklist-strict
```

**Local CRC:**

```bash
make deploy-local-full
make demo-checklist-local
```

Open the UI:

```bash
oc get route streamlit-ui -n retail-ai-demo
```

---

## 3. Streamlit UI: mock vs live data

The sidebar links to several pages. **Know which are real before you present** — especially if someone asks about the Dashboard charts.

| Page | Data | Presenter note |
|------|------|----------------|
| **app** (main chat) | **Live** | Primary demo surface. Multi-agent queries use LLM, RAG, Search, and Analytics MCP against Postgres (when configured). |
| **app → Configuration tab** | **Live (read-only)** | Shows **current environment settings** from `.env` / ConfigMap (MCP URLs, model name, fallbacks, redacted secrets). Does not edit config — changes require `.env` updates and a restart. |
| **Dashboard** | **Mostly mock** | Query Analytics charts, Business Insights cards/table, and Alerts use **synthetic demo data** for visual impact. System Overview metrics are largely static/random. Agent Performance can show **in-session** agent counters if the chat app was used first in the same browser session — not historical Postgres data. |
| **Agents** | **Mixed** | Live agent status when the crew is initialized; otherwise falls back to demo metrics. |
| **Analytics** | **Live** | Inventory, customer, and product queries via Analytics MCP → PostgreSQL. Use this page to show **real retail data**. |
| **MCP Tools** | **Live** | Direct MCP health checks and tool invocations (LLM, RAG, Search, Analytics). |

**Suggested talking point:** *"The Dashboard illustrates operational monitoring for executives; the chat and Analytics pages show the live AI and data layer underneath."*

**Do not claim** Dashboard query volumes, business insight rows, or alert notifications are pulled from production telemetry — they are intentional placeholders for the demo narrative.

**Configuration tab:** Safe to show during a demo when explaining how the stack is wired (Ollama vs kServe, Postgres vs JSON fallback, Tavily configured or not). Point out that `TAVILY_API_KEY` and `LLM_API_KEY` appear as `configured (redacted)` when set.

---

## 4. Scenario walkthroughs

### Scenario 1: Strategic fashion trend analysis

**Talking points:**

- Strategic planning for Cape Town stores and the upcoming winter season
- The system understands brands, demographics, and stock — not generic trend lists

**Query:**

```
What winter fashion trends should our Cape Town stores focus on for professional women?
```

**Highlight:** HomeAgent coordinates Trend, Inventory, and Pricing agents. Expect references to trends like "Power Suiting" and "Luxe Knitwear" with inventory and pricing context.

---

### Scenario 2: Personalized cross-sell opportunity

**Talking points:**

- Real-time customer interaction and upsell
- Sarah Johnson just bought a winter coat

**Query:**

```
Customer Sarah Johnson bought a winter coat. What should we recommend?
```

**Highlight:** RAG and analytics drive personalized recommendations aligned with loyalty tier and brand preference — not random accessories.

---

### Scenario 3: Dynamic inventory optimization

**Talking points:**

- Operational supply chain balance across Johannesburg stores

**Query:**

```
Optimize inventory for the upcoming summer season across all Johannesburg stores.
```

**Highlight:** InventoryAgent and TrendAgent collaborate on forecasts, reorder quantities, and redistribution.

---

### Scenario 4: Complex customer complaint resolution

**Talking points:**

- High-value customer retention under pressure

**Query:**

```
A high-value customer is complaining about a delayed delivery and poor service.
```

**Highlight:** Customer, Inventory, and Pricing agents contribute to an empathetic, actionable resolution with retention offer.

---

## 5. If something goes wrong during the demo

| Issue | Quick fix |
|-------|-----------|
| Blank or slow UI | `oc get pods -n retail-ai-demo` — wait for Streamlit Ready |
| Generic RAG answers | Check Milvus priming: `oc logs job/milvus-primer-job -n retail-ai-demo` |
| Search seems fake | Verify `TAVILY_API_KEY` in secret (not `replace-me`) |
| LLM errors | Check InferenceService: `oc get inferenceservice granite-vllm -n retail-ai-demo` |
| Need logs live | `oc logs deployment/streamlit-ui -n retail-ai-demo -f` |

Re-run validation:

```bash
make validate
```

---

## 6. Related documentation

- [DEPLOYMENT.md](DEPLOYMENT.md) — full deploy pipeline
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design
- [README.md](../README.md) — quick start and Makefile targets
