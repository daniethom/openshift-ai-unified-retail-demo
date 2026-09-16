# Changelog

All notable changes to the Meridian Retail AI demo are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Phase 1 — Central configuration
- `config/settings.py` as the single source of truth for environment variables
- Unified MCP URL variables: `LLM_MCP_URL`, `RAG_MCP_URL`, `SEARCH_MCP_URL`, `ANALYTICS_MCP_URL`
- Unified LLM settings: `LLM_API_BASE`, `LLM_API_KEY`, `LLM_MODEL_NAME` (with `GRANITE_ENDPOINT` legacy alias)
- Milvus and RAG settings: `MILVUS_URI`, `MILVUS_COLLECTION_NAME`, `RAG_USE_FALLBACK`, `EMBEDDING_MODEL`
- Updated `.env.example` with the full configuration contract
- `config` package registered in `pyproject.toml`

#### Phase 2 — MCP HTTP client layer
- `agents/mcp_client.py` — HTTP client for all MCP `/invoke` endpoints
- `agents/tools/rag_tools.py` — `MCPRagRetriever` backed by the RAG MCP server
- Updated `agents/tools/search_tools.py` — `TrendSearcher` uses Tavily via MCP client
- Agents updated to call MCP over HTTP: `pricing_agent`, `inventory_agent`, `trend_agent`
- `streamlit_app/app.py` reads MCP URLs from `config/settings.py`
- `agents/__init__.py` package init

#### Phase 3 — Real backends and OpenShift AI manifests
- `rag/service.py` — Milvus indexing and retrieval from JSON data files
- `mcp_servers/rag_server.py` wired to Milvus (with graceful fallback)
- `mcp_servers/search_server.py` wired to Tavily API (with graceful fallback)
- `mcp_servers/analytics_server.py` — added `get_product_details` and `get_demand_analytics` tools
- `scripts/prime_database.py` — loads products, trends, and market data into Milvus
- Milvus standalone deployment (`k8s/base/milvus-*.yaml`)
- kServe `InferenceService` for Granite/vLLM (production overlay)
- OpenShift Routes for Streamlit UI and Granite LLM API
- Milvus priming Job with init container waiting for Milvus health
- Health probes (`/healthz`) on RAG and Search MCP deployments

#### Phase 4 — CI/CD, model download, and operational tooling
- `scripts/build-image.sh` — build and push to OpenShift internal registry
- OpenShift `BuildConfig` and `ImageStream` (production overlay)
- `scripts/download_model.py` and `scripts/download-model.sh` — Granite model download (local or cluster Job)
- `k8s/overlays/production/model-download-job.yaml` — populates `granite-model-storage` PVC
- `scripts/validate-deployment.sh` — automated pre-demo health checks
- `scripts/demo-day-checklist.sh` — routes, pod status, sample queries, readiness summary
- `scripts/lib/common.sh` — shared shell helpers for deploy scripts
- GitHub Actions CI (`.github/workflows/ci.yml`) — lint, test, Docker build
- `.dockerignore` for leaner container images
- Makefile targets: `deploy-*-full`, `build-images*`, `download-model*`, `validate*`, `demo-checklist*`
- `[model]` optional dependency (`huggingface_hub`) in `pyproject.toml`

#### Documentation
- Rewrote `README.md`
- Added `docs/DEPLOYMENT.md`
- Updated `docs/ARCHITECTURE.md`, `docs/MCP_INTEGRATION.md`, `docs/MULTI_AGENT_DESIGN.md`, `docs/DEMO_GUIDE.md`
- Updated `CONTRIBUTING.md`

#### Tests
- `tests/unit/test_settings.py`
- `tests/unit/test_mcp_client.py`
- `tests/unit/test_rag_service.py`
- `tests/unit/test_download_model.py`
- Updated `test_trend_agent.py`, `test_rag_server.py`, `test_search_server.py`

#### Phase 5 — Test coverage and PostgreSQL data layer
- Stabilized unit/integration tests; CI enforces 75% coverage on core packages
- `db/` package with SQLAlchemy models, Alembic migrations, JSON seed loader, repository layer
- `DATABASE_URL`, `USE_JSON_FALLBACK`, and `DATA_PATH` in `config/settings.py`
- Analytics MCP reads PostgreSQL via `db/service.py` with JSON fallback; added customer tools
- Customer, inventory, and pricing agents use analytics MCP / real product IDs (`MF001`, etc.)
- `rag/service.py` can build Milvus documents from PostgreSQL when configured
- OpenShift PostgreSQL PVC/deployment/service plus `db-migrate-job` and `db-seed-job`
- `scripts/deploy-openshift.sh` flags: `--migrate-db`, `--seed-db`
- Makefile targets: `migrate-db`, `seed-db`, `test-cov`
- Streamlit smoke tests; legacy `rag/knowledge_base.py` and `rag/retriever.py` marked deprecated

### Changed

#### Phase 1 — Configuration
- Replaced scattered hardcoded `localhost` URLs with environment-driven configuration
- OpenShift ConfigMaps use in-cluster service DNS names (e.g. `http://llm-server:8001`)
- Production overlay patches `LLM_API_BASE` to kServe in-cluster endpoint

#### Phase 2 — Agent integration
- Agents no longer expect Python mock objects with `.search()` / `.retrieve_documents()` methods
- `agents/home_agent.py` imports real `BaseAgent` from `agents/base_agent.py` (removed placeholder class)
- `TrendAgent` constructor accepts optional `rag_retriever` and uses `MCPRagRetriever` by default

#### Phase 3 — Kubernetes layout
- Consolidated deployments and services under `k8s/base/`
- Kustomize overlays: `local` (resource limits, no kServe) and `production` (kServe, BuildConfig, model PVC)
- Standardized namespace and secret names to `retail-ai-demo` / `mcp-servers-secrets`
- `Dockerfile` installs production + model dependencies (not dev extras)

#### Phase 4 — Deploy scripts
- `scripts/deploy-local.sh` and `scripts/deploy-openshift.sh` support `--build`, `--validate`, `--skip-prime`, and related flags
- `scripts/run-tests.sh` sets `RAG_USE_FALLBACK=true` for consistent local/CI testing
- `scripts/setup.sh` references `make deploy-prod-full` and demo checklist targets

### Fixed
- Kustomize resource paths (all manifests under `k8s/base/` tree for valid builds)
- Deploy script paths to priming Job and Kustomize overlays
- Inconsistent namespaces (`meridian-demo` vs `retail-ai-demo`) in priming Job
- Mismatched secret names (`meridian-secrets` vs `mcp-servers-secrets`)
- `LLM_API_BASE` vs `GRANITE_ENDPOINT` env var confusion in LLM MCP server
- Production overlay now includes `deployment-patch.yaml`
- Local overlay `configmap-patch.yaml` was a duplicate kustomization file (corrected)

### Deprecated
- `GRANITE_ENDPOINT` — use `LLM_API_BASE` instead (alias still supported)
- `k8s/configmaps/*.yaml` — superseded by `k8s/base/configmap.yaml` and `streamlit-ui-config.yaml`
- `agents/crew_config.yaml` MCP endpoint block — reference only; runtime uses `config/settings.py`

---

## [1.0.0-a1] — Initial demo release

- Multi-agent CrewAI orchestration (Home, Trend, Inventory, Pricing, Customer agents)
- MCP server skeletons (LLM, RAG, Search, Analytics)
- Streamlit demo UI with synthetic JSON data
- Basic Kustomize manifests and local/production overlay stubs
- Simulated RAG and search responses for conference demos
- Pytest unit and integration test suites

[Unreleased]: https://github.com/daniethom/openshift-ai-unified-demo/compare/v1.0.0-a1...HEAD
[1.0.0-a1]: https://github.com/daniethom/openshift-ai-unified-demo/releases/tag/v1.0.0-a1
