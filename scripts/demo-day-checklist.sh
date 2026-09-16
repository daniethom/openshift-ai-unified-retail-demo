#!/bin/bash
# Demo-day checklist: routes, pod status, jobs, and sample queries in one view.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

NAMESPACE="${OPENSHIFT_NAMESPACE:-retail-ai-demo}"
RUN_VALIDATE=false
STRICT=false
SKIP_INFERENCE=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Print a demo-day readiness report for presenters and operators.

Options:
  --namespace NAME     OpenShift namespace (default: retail-ai-demo)
  --validate           Also run scripts/validate-deployment.sh
  --strict             Exit with code 1 if any checklist item fails
  --skip-inference     Skip kServe checks (useful for local/CRC)
  -h, --help           Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --namespace) NAMESPACE="$2"; shift 2 ;;
        --validate) RUN_VALIDATE=true; shift ;;
        --strict) STRICT=true; shift ;;
        --skip-inference) SKIP_INFERENCE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) error "Unknown option: $1" ;;
    esac
done

PROJECT_ROOT="$(script_root)"
ISSUES=0

section() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

check_item() {
    local label="$1"
    local ok="$2"
    if [[ "${ok}" == "true" ]]; then
        success "[✓] ${label}"
    else
        warn "[✗] ${label}"
        ISSUES=$((ISSUES + 1))
    fi
}

ensure_openshift_login

section "Meridian Retail AI — Demo Day Checklist"
info "Namespace: ${NAMESPACE}"
info "Cluster:   $(oc whoami --show-server 2>/dev/null || echo 'unknown')"
info "User:      $(oc whoami 2>/dev/null || echo 'unknown')"
info "Time:      $(date)"

section "1. Pre-demo manual checks"
echo "  [ ] Secrets updated (not placeholder 'replace-me'):"
echo "        oc get secret mcp-servers-secrets -n ${NAMESPACE}"
echo "  [ ] Tavily API key is valid and has quota remaining"
echo "  [ ] Granite model downloaded (production only):"
echo "        oc get job granite-model-download -n ${NAMESPACE}"
echo "  [ ] Milvus priming completed:"
echo "        oc get job milvus-primer-job -n ${NAMESPACE}"
echo "  [ ] Browser bookmarked to Streamlit route (see section 3)"
echo "  [ ] Backup demo queries copied (see section 6)"

section "2. Pod status"
if oc get namespace "${NAMESPACE}" >/dev/null 2>&1; then
    oc get pods -n "${NAMESPACE}" -o wide
    echo ""
    for deploy in llm-server rag-server search-server analytics-server streamlit-ui milvus-standalone; do
        if ! oc get deployment "${deploy}" -n "${NAMESPACE}" >/dev/null 2>&1; then
            check_item "Deployment ${deploy} exists" false
            continue
        fi
        desired="$(oc get deployment "${deploy}" -n "${NAMESPACE}" -o jsonpath='{.spec.replicas}')"
        ready="$(oc get deployment "${deploy}" -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}')"
        ready="${ready:-0}"
        [[ "${ready}" == "${desired}" && "${desired}" != "0" ]] \
            && check_item "${deploy} ready (${ready}/${desired})" true \
            || check_item "${deploy} ready (${ready}/${desired})" false
    done
else
    check_item "Namespace ${NAMESPACE} exists" false
fi

section "3. Routes and URLs"
if oc get route streamlit-ui -n "${NAMESPACE}" >/dev/null 2>&1; then
    streamlit_host="$(oc get route streamlit-ui -n "${NAMESPACE}" -o jsonpath='{.spec.host}')"
    streamlit_url="https://${streamlit_host}"
    success "Streamlit UI: ${streamlit_url}"
    if curl -ksf "${streamlit_url}" >/dev/null 2>&1; then
        check_item "Streamlit route responds over HTTPS" true
    else
        check_item "Streamlit route responds over HTTPS" false
    fi
else
    check_item "Route streamlit-ui exists" false
    streamlit_url="(not available — run: oc get route -n ${NAMESPACE})"
fi

if [[ "${SKIP_INFERENCE}" == "false" ]] && oc get route granite-vllm -n "${NAMESPACE}" >/dev/null 2>&1; then
    granite_host="$(oc get route granite-vllm -n "${NAMESPACE}" -o jsonpath='{.spec.host}')"
    success "Granite LLM API route: https://${granite_host}"
elif [[ "${SKIP_INFERENCE}" == "false" ]] && oc get inferenceservice granite-vllm -n "${NAMESPACE}" >/dev/null 2>&1; then
    info "Granite InferenceService (in-cluster): http://granite-vllm-predictor.${NAMESPACE}.svc.cluster.local/v1"
else
    info "Granite kServe route not deployed (expected on local/CRC overlays)."
fi

section "4. Background jobs"
for job in milvus-primer-job granite-model-download; do
    if oc get job "${job}" -n "${NAMESPACE}" >/dev/null 2>&1; then
        succeeded="$(oc get job "${job}" -n "${NAMESPACE}" -o jsonpath='{.status.succeeded}')"
        [[ "${succeeded}" == "1" ]] \
            && check_item "Job ${job} completed" true \
            || check_item "Job ${job} completed (check: oc logs job/${job} -n ${NAMESPACE})" false
    else
        info "Job ${job}: not found (may not have been run yet)"
    fi
done

section "5. kServe / LLM status"
if [[ "${SKIP_INFERENCE}" == "true" ]]; then
    info "Skipped (--skip-inference)."
elif oc get inferenceservice granite-vllm -n "${NAMESPACE}" >/dev/null 2>&1; then
    isvc_ready="$(oc get inferenceservice granite-vllm -n "${NAMESPACE}" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)"
    [[ "${isvc_ready}" == "True" ]] \
        && check_item "InferenceService granite-vllm is Ready" true \
        || check_item "InferenceService granite-vllm is Ready (current: ${isvc_ready:-Unknown})" false
    oc get inferenceservice granite-vllm -n "${NAMESPACE}" 2>/dev/null || true
else
    info "InferenceService granite-vllm not found."
fi

section "6. Sample demo queries (paste into Streamlit)"
cat <<'EOF'
  Scenario 1 — Fashion trends:
    What winter fashion trends should our Cape Town stores focus on for professional women?

  Scenario 2 — Cross-sell:
    Customer Sarah Johnson bought a winter coat. What should we recommend?

  Scenario 3 — Inventory optimization:
    Optimize inventory for the upcoming summer season across all Johannesburg stores.

  Scenario 4 — Customer complaint:
    A high-value customer is complaining about a delayed delivery and poor service.
EOF

section "7. Useful commands during the demo"
cat <<EOF
  Watch pods:        oc get pods -n ${NAMESPACE} -w
  Streamlit logs:    oc logs deployment/streamlit-ui -n ${NAMESPACE} -f
  LLM MCP logs:      oc logs deployment/llm-server -n ${NAMESPACE} -f
  RAG MCP logs:      oc logs deployment/rag-server -n ${NAMESPACE} -f
  Re-run validation: make validate
  Full re-deploy:    make deploy-prod-full
EOF

if [[ "${RUN_VALIDATE}" == "true" ]]; then
    section "8. Automated validation"
    if [[ "${SKIP_INFERENCE}" == "true" ]]; then
        "${PROJECT_ROOT}/scripts/validate-deployment.sh" --namespace "${NAMESPACE}" --skip-inference || ISSUES=$((ISSUES + 1))
    else
        "${PROJECT_ROOT}/scripts/validate-deployment.sh" --namespace "${NAMESPACE}" || ISSUES=$((ISSUES + 1))
    fi
fi

section "Summary"
if [[ "${ISSUES}" -eq 0 ]]; then
    success "Demo checklist: READY (${ISSUES} issues)"
    echo ""
    success "Open the demo at: ${streamlit_url}"
    exit 0
fi

warn "Demo checklist: NOT READY (${ISSUES} issue(s) found)"
echo ""
info "Fix issues above, then re-run: make demo-checklist"
[[ "${STRICT}" == "true" ]] && exit 1
exit 0
