#!/bin/bash
# Deploy the Meridian Retail AI Demo to a local OpenShift (CRC) cluster.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

PROJECT_ROOT="$(script_root)"
NAMESPACE="${OPENSHIFT_NAMESPACE:-retail-ai-demo}"
OVERLAY="${PROJECT_ROOT}/k8s/overlays/local"

DO_BUILD=false
DO_VALIDATE=false
DO_SKIP_PRIME=false
USE_OPENSHIFT_BUILD=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --build               Build and push the application image before deploying
  --openshift-build     Build on-cluster with OpenShift BuildConfig
  --validate            Run post-deploy validation checks
  --skip-prime          Skip the Milvus priming Job
  --namespace NAME      OpenShift namespace (default: retail-ai-demo)
  -h, --help            Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build) DO_BUILD=true; shift ;;
        --openshift-build) DO_BUILD=true; USE_OPENSHIFT_BUILD=true; shift ;;
        --validate) DO_VALIDATE=true; shift ;;
        --skip-prime) DO_SKIP_PRIME=true; shift ;;
        --namespace) NAMESPACE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) error "Unknown option: $1" ;;
    esac
done

ensure_openshift_login
ensure_namespace "${NAMESPACE}"

if [[ "${DO_BUILD}" == "true" ]]; then
    info "Building application image..."
    if [[ "${USE_OPENSHIFT_BUILD}" == "true" ]]; then
        "${PROJECT_ROOT}/scripts/build-image.sh" --openshift-build --namespace "${NAMESPACE}"
    else
        "${PROJECT_ROOT}/scripts/build-image.sh" --namespace "${NAMESPACE}"
    fi
else
    warn "Skipping image build. Use --build or ensure the image already exists in the cluster registry."
fi

info "Applying Kustomize overlay for 'local' environment..."
oc apply -k "${OVERLAY}" -n "${NAMESPACE}" || error "Failed to apply local overlay."

if [[ "${DO_SKIP_PRIME}" == "false" ]]; then
    info "Applying Milvus priming job..."
    oc delete job milvus-primer-job -n "${NAMESPACE}" --ignore-not-found
    oc apply -f "${PROJECT_ROOT}/k8s/base/priming-job.yaml" -n "${NAMESPACE}"
    info "Watch priming with: oc logs -n ${NAMESPACE} job/milvus-primer-job -f"
else
    warn "Skipping Milvus priming job."
fi

if [[ "${DO_VALIDATE}" == "true" ]]; then
    info "Running deployment validation..."
    "${PROJECT_ROOT}/scripts/validate-deployment.sh" --namespace "${NAMESPACE}" --skip-inference
fi

success "Deployment manifests applied to namespace '${NAMESPACE}'."
info "Monitor pods with: oc get pods -n ${NAMESPACE} -w"
