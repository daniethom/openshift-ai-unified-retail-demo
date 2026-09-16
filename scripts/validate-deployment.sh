#!/bin/bash
# Pre-demo validation for the Meridian Retail AI deployment on OpenShift.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

NAMESPACE="${OPENSHIFT_NAMESPACE:-retail-ai-demo}"
CHECK_INFERENCE=true
CHECK_JOBS=true
FAILURES=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --namespace NAME        OpenShift namespace (default: retail-ai-demo)
  --skip-inference        Skip kServe InferenceService checks
  --skip-jobs             Skip completed Job checks
  -h, --help              Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --namespace) NAMESPACE="$2"; shift 2 ;;
        --skip-inference) CHECK_INFERENCE=false; shift ;;
        --skip-jobs) CHECK_JOBS=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) error "Unknown option: $1" ;;
    esac
done

pass() { success "PASS: $1"; }
fail() { FAILURES=$((FAILURES + 1)); warn "FAIL: $1"; }

ensure_openshift_login
info "Validating deployment in namespace '${NAMESPACE}'..."

if ! oc get namespace "${NAMESPACE}" >/dev/null 2>&1; then
    fail "Namespace '${NAMESPACE}' does not exist."
else
    pass "Namespace '${NAMESPACE}' exists."
fi

for deploy in llm-server rag-server search-server analytics-server streamlit-ui milvus-standalone; do
    if ! oc get deployment "${deploy}" -n "${NAMESPACE}" >/dev/null 2>&1; then
        fail "Deployment '${deploy}' not found."
        continue
    fi

    desired="$(oc get deployment "${deploy}" -n "${NAMESPACE}" -o jsonpath='{.spec.replicas}')"
    ready="$(oc get deployment "${deploy}" -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}')"
    ready="${ready:-0}"

    if [[ "${ready}" == "${desired}" && "${desired}" != "0" ]]; then
        pass "Deployment '${deploy}' is ready (${ready}/${desired})."
    else
        fail "Deployment '${deploy}' is not ready (${ready}/${desired})."
    fi
done

if oc get secret mcp-servers-secrets -n "${NAMESPACE}" >/dev/null 2>&1; then
    tavily_key="$(oc get secret mcp-servers-secrets -n "${NAMESPACE}" -o jsonpath='{.data.TAVILY_API_KEY}' 2>/dev/null || true)"
    placeholder="$(printf 'replace-me' | base64 | tr -d '\n')"
    if [[ -z "${tavily_key}" || "${tavily_key}" == "${placeholder}" ]]; then
        fail "Secret mcp-servers-secrets still uses placeholder TAVILY_API_KEY."
    else
        pass "Secret mcp-servers-secrets is configured."
    fi
else
    fail "Secret mcp-servers-secrets not found."
fi

if oc get route streamlit-ui -n "${NAMESPACE}" >/dev/null 2>&1; then
    route_host="$(oc get route streamlit-ui -n "${NAMESPACE}" -o jsonpath='{.spec.host}')"
    if curl -ksf "https://${route_host}" >/dev/null 2>&1; then
        pass "Streamlit route is reachable: https://${route_host}"
    else
        fail "Streamlit route exists but is not reachable yet: https://${route_host}"
    fi
else
    fail "Route streamlit-ui not found."
fi

info "Running in-cluster MCP health checks..."
read -r -d '' HEALTH_SCRIPT <<'PY' || true
import os
import sys
import urllib.error
import urllib.request

checks = {
    "llm-server": os.environ["LLM_MCP_URL"] + "/healthz",
    "rag-server": os.environ["RAG_MCP_URL"] + "/healthz",
    "search-server": os.environ["SEARCH_MCP_URL"] + "/healthz",
    "analytics-server": os.environ["ANALYTICS_MCP_URL"] + "/healthz",
}

failed = False
for name, url in checks.items():
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            if response.status >= 400:
                print(f"FAIL {name} {url} status={response.status}")
                failed = True
            else:
                print(f"PASS {name} {url}")
    except urllib.error.URLError as exc:
        print(f"FAIL {name} {url} error={exc}")
        failed = True

sys.exit(1 if failed else 0)
PY

if oc get deployment streamlit-ui -n "${NAMESPACE}" >/dev/null 2>&1; then
    health_output="$(oc exec deployment/streamlit-ui -n "${NAMESPACE}" -- env \
        LLM_MCP_URL="http://llm-server:8001" \
        RAG_MCP_URL="http://rag-server:8002" \
        SEARCH_MCP_URL="http://search-server:8003" \
        ANALYTICS_MCP_URL="http://analytics-server:8004" \
        python -c "${HEALTH_SCRIPT}" 2>&1 || true)"

    while IFS= read -r line; do
        if [[ "${line}" == PASS* ]]; then
            pass "${line#PASS: }"
        elif [[ "${line}" == FAIL* ]]; then
            fail "${line#FAIL: }"
        fi
    done <<< "${health_output}"
else
    fail "Cannot run MCP health checks because streamlit-ui is unavailable."
fi

if [[ "${CHECK_JOBS}" == "true" ]]; then
    for job in milvus-primer-job granite-model-download; do
        if ! oc get job "${job}" -n "${NAMESPACE}" >/dev/null 2>&1; then
            warn "Job '${job}' not found (may not have been run yet)."
            continue
        fi

        complete="$(oc get job "${job}" -n "${NAMESPACE}" -o jsonpath='{.status.succeeded}')"
        if [[ "${complete}" == "1" ]]; then
            pass "Job '${job}' completed successfully."
        else
            fail "Job '${job}' has not completed successfully."
        fi
    done
fi

if [[ "${CHECK_INFERENCE}" == "true" ]]; then
    if oc get inferenceservice granite-vllm -n "${NAMESPACE}" >/dev/null 2>&1; then
        ready="$(oc get inferenceservice granite-vllm -n "${NAMESPACE}" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)"
        if [[ "${ready}" == "True" ]]; then
            pass "InferenceService granite-vllm is Ready."
        else
            fail "InferenceService granite-vllm is not Ready yet."
        fi
    else
        warn "InferenceService granite-vllm not found (skipped for local overlays)."
    fi
fi

echo ""
if [[ "${FAILURES}" -gt 0 ]]; then
    error "Validation failed with ${FAILURES} issue(s)."
fi

success "All validation checks passed. The demo is ready."
