#!/usr/bin/env bash
# Start all four MCP servers for local development.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PYTHON="${ROOT}/.venv/bin/python"
UVICORN="${ROOT}/.venv/bin/uvicorn"

if [[ ! -x "$UVICORN" ]]; then
  echo "Missing virtualenv. Run: make install" >&2
  exit 1
fi

start_server() {
  local module="$1"
  local port="$2"
  if lsof -i ":${port}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port ${port} already in use (${module})"
    return 0
  fi
  echo "Starting ${module} on port ${port}..."
  nohup "$UVICORN" "${module}:app" --host 127.0.0.1 --port "${port}" --log-level warning \
    > "${ROOT}/.local/${module//./-}-${port}.log" 2>&1 &
}

mkdir -p "${ROOT}/.local"

start_server "mcp_servers.llm_server" 8001
start_server "mcp_servers.rag_server" 8002
start_server "mcp_servers.search_server" 8003
start_server "mcp_servers.analytics_server" 8004

sleep 2
echo
echo "Health checks:"
for port in 8001 8002 8003 8004; do
  printf "  %s: " "$port"
  curl -sf "http://127.0.0.1:${port}/healthz" || echo "not ready"
  echo
done
