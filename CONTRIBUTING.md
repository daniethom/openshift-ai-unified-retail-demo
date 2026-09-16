# Contributing to the Meridian Retail AI Demo

Thank you for contributing. This document covers setup, standards, testing, and the pull request process.

## Getting started

1. **Fork and clone** the repository.
2. **Install dependencies:**

   ```bash
   make install
   ```

   Installs `[dev,model]` extras (pytest, ruff, black, huggingface_hub).

3. **Configure environment:**

   ```bash
   cp .env.example .env
   ```

   Set at minimum `TAVILY_API_KEY`. Use `RAG_USE_FALLBACK=true` if Milvus is not running locally.

4. **Optional — interactive setup:**

   ```bash
   ./scripts/setup.sh
   ```

## Development workflow

Create a feature branch:

```bash
git checkout -b feature/your-feature-name
```

Keep commits focused with clear messages.

## Project conventions

### Configuration

- Add new environment variables to **`config/settings.py`** and **`.env.example`**
- Patch OpenShift values in **`k8s/base/configmap.yaml`** and overlays — do not hardcode URLs in application code
- Legacy names (e.g. `GRANITE_ENDPOINT`) should remain as aliases when renamed

### MCP integration

- Agents call MCP servers only through **`agents/mcp_client.py`**
- New tools: add handler in the relevant `mcp_servers/*.py` file and a client function in `mcp_client.py`

### Kubernetes

- Base manifests: `k8s/base/`
- Environment-specific patches: `k8s/overlays/local/` or `production/`
- Validate with: `kubectl kustomize k8s/overlays/local`

## Code standards

- **Lint:** Ruff (`make lint`)
- **Format:** Black (`make format`)

Before committing:

```bash
make format
make lint
make test
```

## Testing

```bash
make test                    # Full suite (sets RAG_USE_FALLBACK=true)
pytest tests/unit            # Unit tests only
pytest tests/integration       # Integration tests
```

Add tests for new behavior:

- MCP servers: `tests/unit/test_*_server.py`
- MCP client: `tests/unit/test_mcp_client.py`
- Settings: `tests/unit/test_settings.py`
- RAG: `tests/unit/test_rag_service.py`

CI (`.github/workflows/ci.yml`) runs lint, test, and Docker build on push/PR.

## OpenShift and deployment changes

If you change deployment manifests or scripts:

1. Verify Kustomize builds: `kubectl kustomize k8s/overlays/production`
2. Update **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** and **[README.md](README.md)** if commands or flags change
3. Test validation: `make validate-local` (CRC) or document production-only steps

Useful Makefile targets when testing deploy changes:

```bash
make build-images-local
make deploy-local-full
make demo-checklist-local
```

## Pull request process

1. Push your branch to your fork.
2. Open a PR against `main`.
3. Include a clear title and summary; reference issues (e.g. "Closes #123").

### PR checklist

- [ ] Code follows project style (`make format`, `make lint`)
- [ ] Tests added or updated for the change
- [ ] `make test` passes locally
- [ ] Documentation updated if behavior, config, or deploy steps changed
- [ ] No secrets or `.env` files committed

## Documentation

When changing architecture, MCP contracts, or deployment:

| Change type | Update |
|-------------|--------|
| User-visible features or fixes | `CHANGELOG.md` (under `[Unreleased]`) |
| Config / env vars | `.env.example`, `docs/DEPLOYMENT.md`, README configuration table |
| MCP API | `docs/MCP_INTEGRATION.md` |
| Agents | `docs/MULTI_AGENT_DESIGN.md` |
| Demo flow | `docs/DEMO_GUIDE.md` |
| System design | `docs/ARCHITECTURE.md` |

Thank you for helping improve the demo.
