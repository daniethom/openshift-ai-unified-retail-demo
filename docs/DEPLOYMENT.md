# Deployment Guide

This guide covers repeatable deployment of the Meridian Retail AI demo on OpenShift Local (CRC) and production OpenShift AI clusters.

## Prerequisites

| Requirement | Local (CRC) | Production |
|-------------|-------------|------------|
| OpenShift 4.x | CRC or similar | OpenShift AI cluster |
| `oc` CLI | Required | Required |
| Container builder | podman or docker | podman, docker, or cluster BuildConfig |
| GPU nodes | Optional | Required for kServe Granite |
| Tavily API key | Required | Required |
| Hugging Face token | Optional | Optional (gated models) |

## Namespace and secrets

Default namespace: `retail-ai-demo`

Create or update secrets **before** deploying (do not commit real keys):

```bash
oc create secret generic mcp-servers-secrets \
  --from-literal=TAVILY_API_KEY='your_tavily_key' \
  --from-literal=LLM_API_KEY='not-needed' \
  --from-literal=HF_TOKEN='your_hf_token_if_required' \
  -n retail-ai-demo \
  --dry-run=client -o yaml | oc apply -f -
```

Replace placeholder values in `k8s/base/secret.yaml` if applying manifests without the command above.

## Deployment paths

### Local / CRC (no kServe)

```bash
make deploy-local-full
```

Equivalent:

```bash
./scripts/deploy-local.sh --build --validate
```

Uses overlay `k8s/overlays/local/` with reduced Milvus/RAG resources.

### Production OpenShift AI (with kServe)

```bash
make deploy-prod-full
```

Equivalent:

```bash
./scripts/deploy-openshift.sh --build --download-model --validate
```

Uses overlay `k8s/overlays/production/` which adds:

- `InferenceService` (Granite via vLLM)
- `BuildConfig` + `ImageStream`
- Model storage PVC
- Granite OpenShift Route

### Deploy script options

Both `deploy-local.sh` and `deploy-openshift.sh` support:

| Flag | Description |
|------|-------------|
| `--build` | Build and push application image |
| `--openshift-build` | Build on-cluster via BuildConfig |
| `--download-model` | Run Granite model download Job (production script only) |
| `--validate` | Run post-deploy validation |
| `--skip-prime` | Skip Milvus priming Job |
| `--namespace NAME` | Override namespace |

## Image build

### Push from workstation

```bash
make build-images
# or
./scripts/build-image.sh --namespace retail-ai-demo
```

Build locally without pushing:

```bash
make build-images-local
```

### Build on OpenShift

```bash
make build-images-openshift
# or
./scripts/build-image.sh --openshift-build
```

Image reference used in manifests:

```
image-registry.openshift-image-registry.svc:5000/retail-ai-demo/meridian-retail-ai:latest
```

## Milvus and RAG priming

After deploy, the Milvus priming Job loads JSON data into the `meridian_knowledge` collection:

```bash
oc delete job milvus-primer-job -n retail-ai-demo --ignore-not-found
oc apply -f k8s/base/priming-job.yaml
oc logs -n retail-ai-demo job/milvus-primer-job -f
```

Deploy scripts run this automatically unless `--skip-prime` is set.

Local priming (without cluster):

```bash
python scripts/prime_database.py --recreate
```

## Granite model download

Default model: `ibm-granite/granite-3.0-2b-instruct` (configure via `MODEL_HF_REPO`).

### On cluster

```bash
make download-model-cluster
```

Writes weights to PVC `granite-model-storage` at path `granite-3b` (matches InferenceService `storageUri`).

### Locally

```bash
make download-model
# or
pip install -e ".[model]"
python scripts/download_model.py --output ./models/granite-3b
```

## kServe InferenceService

Production manifest: `k8s/overlays/production/inferenceservice-granite.yaml`

The LLM MCP server calls the in-cluster OpenAI-compatible endpoint:

```
http://granite-vllm-predictor.retail-ai-demo.svc.cluster.local/v1
```

Patched via `k8s/overlays/production/configmap-patch.yaml`.

**Before demo:** confirm GPU nodes, model PVC is populated, and InferenceService is Ready:

```bash
oc get inferenceservice granite-vllm -n retail-ai-demo
```

Adjust `storageUri`, runtime, or resource limits in the InferenceService manifest to match your OpenShift AI version.

## Routes

| Route | Service | Purpose |
|-------|---------|---------|
| `streamlit-ui` | Streamlit UI | Main demo interface |
| `granite-vllm` | kServe predictor | External LLM API (production) |

```bash
oc get route -n retail-ai-demo
```

## Validation and demo checklist

### Automated validation

```bash
make validate              # Production (includes kServe)
make validate-local        # Skip InferenceService checks
```

Checks: deployments ready, secrets configured, Streamlit route reachable, MCP `/healthz` endpoints, completed Jobs.

### Demo-day checklist

```bash
make demo-checklist              # Human-readable status report
make demo-checklist-strict       # Fails if anything is not ready
make demo-checklist-local        # CRC without kServe checks
```

See [DEMO_GUIDE.md](DEMO_GUIDE.md) for presenter script and sample queries.

## Kustomize layout

```
k8s/base/                    # Shared resources
  namespace.yaml
  configmap.yaml             # MCP URLs, Milvus, LLM settings
  secret.yaml                # Template secrets
  *-deployment.yaml          # MCP servers + Streamlit
  milvus-*.yaml              # Milvus standalone + PVC
  route-streamlit.yaml
  priming-job.yaml           # Applied separately by deploy scripts

k8s/overlays/local/          # Smaller resources, no kServe
k8s/overlays/production/     # kServe, BuildConfig, model PVC, Granite route
```

Apply manually:

```bash
kubectl kustomize k8s/overlays/local | oc apply -f -
kubectl kustomize k8s/overlays/production | oc apply -f -
```

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) on push/PR:

1. Install dependencies (`[dev,model]`)
2. Ruff + Black
3. Pytest with `RAG_USE_FALLBACK=true`
4. Docker image build and import smoke test

## Troubleshooting

| Symptom | Check |
|---------|--------|
| Pods `ImagePullBackOff` | Run `make build-images` or verify image exists in internal registry |
| RAG returns fallback docs | Milvus not ready or priming Job failed — check `oc logs job/milvus-primer-job` |
| Search returns fallback | `TAVILY_API_KEY` missing or invalid |
| LLM errors | InferenceService not Ready or wrong `LLM_API_BASE` in ConfigMap |
| Priming Job fails | RAG pod needs ~2Gi memory for embedding model; check Milvus health on `:9091/healthz` |

Useful commands:

```bash
oc get pods -n retail-ai-demo -w
oc logs deployment/streamlit-ui -n retail-ai-demo -f
oc logs deployment/llm-server -n retail-ai-demo -f
make demo-checklist
```

Remove all cluster resources:

```bash
./scripts/cleanup.sh
```

## Local development without OpenShift

1. `make install && cp .env.example .env`
2. Start four MCP servers (ports 8001–8004)
3. Optional: Milvus + `python scripts/prime_database.py --recreate`
4. `make run-ui`

Set in `.env`:

```bash
RAG_USE_FALLBACK=true   # Skip Milvus for UI-only testing
USE_JSON_FALLBACK=true  # Read data/*.json instead of PostgreSQL
# Local LLM via Ollama (see "Local LLM with Ollama" below)
LLM_API_BASE=http://127.0.0.1:11434/v1
LLM_MODEL_NAME=granite3.2:8b
LLM_API_KEY=ollama
```

For containerized PostgreSQL on a Podman workstation, see [Local development with Podman](#local-development-with-podman) below.

## Local development with Podman

This project targets **Podman Desktop** and **`podman compose`** for local containers. GitHub Actions CI still uses Docker; the same `Dockerfile` and `compose.yml` work with both runtimes.

### Prerequisites

| Tool | Purpose |
|------|---------|
| [Podman Desktop](https://podman-desktop.io/) | Local container engine |
| `podman compose` | Compose support (built into Podman Desktop) |
| Python 3.11+ | App, MCP servers, migrations |

### Docker vs Podman command map

| Docker | Podman |
|--------|--------|
| `docker compose up -d` | `podman compose up -d` |
| `docker run …` | `podman run …` |
| `docker build -t …` | `podman build -t …` |
| `docker ps` | `podman ps` |

### PostgreSQL with Compose (recommended)

The repo root includes `compose.yml` for a local Postgres 16 instance:

```bash
podman compose up -d
podman compose ps   # wait until postgres is healthy
```

Add to `.env` (or export in your shell):

```bash
DATABASE_URL="postgresql+asyncpg://meridian:secret@127.0.0.1:5432/meridian"
USE_JSON_FALLBACK=false
```

Initialize schema and seed data from `data/*.json`:

```bash
make migrate-db
make seed-db
```

Stop and remove containers:

```bash
podman compose down        # keep data volume
podman compose down -v     # delete postgres data
```

### PostgreSQL with `podman run` (without Compose)

```bash
podman run -d --name meridian-pg \
  -e POSTGRES_DB=meridian \
  -e POSTGRES_USER=meridian \
  -e POSTGRES_PASSWORD=secret \
  -p 5432:5432 \
  postgres:16
```

Use the same `DATABASE_URL`, `make migrate-db`, and `make seed-db` steps as above.

### Local LLM with Ollama

For local development, run [Ollama](https://ollama.com) instead of kServe. The LLM MCP server (`mcp_servers/llm_server.py`) uses the OpenAI-compatible client, so no code changes are required.

**Recommended model:** `granite3.2:8b` — aligns with the production Granite story and supports RAG/tool-style tasks. For faster (lighter) responses on a laptop, use `granite3.2:2b`.

```bash
# Install Ollama, then pull a model
ollama pull granite3.2:8b

# Verify Ollama is listening
curl -s http://127.0.0.1:11434/
```

Add to `.env`:

```bash
LLM_API_BASE="http://127.0.0.1:11434/v1"
LLM_MODEL_NAME="granite3.2:8b"
LLM_API_KEY="ollama"
```

Restart the LLM MCP server after changing `.env` (port 8001). Quick smoke test:

```bash
curl -sf http://127.0.0.1:8001/healthz
curl -sf -X POST http://127.0.0.1:8001/invoke \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Reply with exactly: Granite is ready."}'
```

**Production vs local:**

| Setting | Local (Ollama) | Production (kServe) |
|---------|----------------|---------------------|
| `LLM_API_BASE` | `http://127.0.0.1:11434/v1` | `http://granite-vllm-predictor…/v1` |
| `LLM_MODEL_NAME` | `granite3.2:8b` (Ollama tag) | `granite-3b` (served name) |
| `LLM_API_KEY` | `ollama` | cluster token or `not-needed` |

### Build and smoke-test the application image

Mirrors the CI `build-image` job:

```bash
podman build -t meridian-retail-ai:local .
podman run --rm meridian-retail-ai:local \
  python -c "import agents, config, db, mcp_servers, rag"
```

### Optional: Milvus for real RAG

With JSON fallback disabled for vectors, run Milvus standalone:

```bash
podman run -d --name milvus \
  -p 19530:19530 -p 9091:9091 \
  docker.io/milvusdb/milvus:v2.4.15 \
  milvus run standalone
```

Then prime the collection:

```bash
export RAG_USE_FALLBACK=false
python scripts/prime_database.py --recreate
```

### Typical Podman + Python workflow

```bash
make install
cp .env.example .env
# Set TAVILY_API_KEY; configure Ollama LLM vars; set DATABASE_URL / USE_JSON_FALLBACK as needed

ollama pull granite3.2:8b        # local LLM (see "Local LLM with Ollama" above)
podman compose up -d             # optional Postgres
make migrate-db && make seed-db  # when using Postgres

make run-mcp-servers             # MCP servers on ports 8001–8004
make run-ui
make test                        # USE_JSON_FALLBACK=true skips DB in CI-style runs
```

## PostgreSQL data layer

Production deployments use in-cluster PostgreSQL as the system of record. JSON files under `data/` are seed input only.

### Local PostgreSQL

Use [Compose](#postgresql-with-compose-recommended) or [`podman run`](#postgresql-with-podman-run-without-compose) as above. Docker equivalents work if you prefer Docker:

```bash
docker compose up -d
# or: docker run -d --name meridian-pg … (same env vars as compose.yml)
```

Set `USE_JSON_FALLBACK=false` in `.env` so MCP servers and agents read from PostgreSQL.

### OpenShift

Base manifests include `postgres-pvc`, `postgres-deployment`, and `postgres-service`. ConfigMap sets `DATABASE_URL`; Secret holds `POSTGRES_PASSWORD` (must match the URL password).

Recommended deploy sequence:

```bash
./scripts/deploy-openshift.sh --build --migrate-db --seed-db --validate
```

Jobs:

- `db-migrate-job` — `alembic upgrade head`
- `db-seed-job` — `python -m db.seed` (idempotent upsert from JSON)
- `milvus-primer-job` — waits for PostgreSQL and Milvus before priming vectors
