#!/bin/bash
# Deploy the Meridian Retail AI Demo to a production OpenShift AI cluster.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

PROJECT_ROOT="$(script_root)"
NAMESPACE="${OPENSHIFT_NAMESPACE:-retail-ai-demo}"
OVERLAY="${PROJECT_ROOT}/k8s/overlays/production"

DO_BUILD=false
DO_DOWNLOAD_MODEL=false
DO_VALIDATE=false
DO_SKIP_PRIME=false
DO_MIGRATE_DB=false
DO_SEED_DB=false
USE_OPENSHIFT_BUILD=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --build               Build and push the application image before deploying
  --openshift-build     Build on-cluster with OpenShift BuildConfig
  --download-model      Run the Granite model download Job after deploy
  --validate            Run post-deploy validation checks
  --migrate-db          Run Alembic database migration Job
  --seed-db             Run database seed Job (loads data/*.json)
  --skip-prime          Skip the Milvus priming Job
  --namespace NAME      OpenShift namespace (default: retail-ai-demo)
  -h, --help            Show this help message

Recommended first production run:
  $(basename "$0") --build --download-model --validate
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build) DO_BUILD=true; shift ;;
        --openshift-build) DO_BUILD=true; USE_OPENSHIFT_BUILD=true; shift ;;
        --download-model) DO_DOWNLOAD_MODEL=true; shift ;;
        --validate) DO_VALIDATE=true; shift ;;
        --migrate-db) DO_MIGRATE_DB=true; shift ;;
        --seed-db) DO_SEED_DB=true; shift ;;
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

info "Applying Kustomize overlay for 'production' environment..."
oc apply -k "${OVERLAY}" -n "${NAMESPACE}" || error "Failed to apply production overlay."

if [[ "${DO_MIGRATE_DB}" == "true" ]]; then
    info "Running database migration job..."
    oc delete job db-migrate-job -n "${NAMESPACE}" --ignore-not-found
    oc apply -f "${PROJECT_ROOT}/k8s/base/db-migrate-job.yaml" -n "${NAMESPACE}"
    oc wait --for=condition=complete job/db-migrate-job -n "${NAMESPACE}" --timeout=300s \
        || error "Database migration job failed."
fi

if [[ "${DO_SEED_DB}" == "true" ]]; then
    info "Running database seed job..."
    oc delete job db-seed-job -n "${NAMESPACE}" --ignore-not-found
    oc apply -f "${PROJECT_ROOT}/k8s/base/db-seed-job.yaml" -n "${NAMESPACE}"
    oc wait --for=condition=complete job/db-seed-job -n "${NAMESPACE}" --timeout=300s \
        || error "Database seed job failed."
fi

if [[ "${DO_SKIP_PRIME}" == "false" ]]; then
    info "Applying Milvus priming job..."
    oc delete job milvus-primer-job -n "${NAMESPACE}" --ignore-not-found
    oc apply -f "${PROJECT_ROOT}/k8s/base/priming-job.yaml" -n "${NAMESPACE}"
    info "Watch priming with: oc logs -n ${NAMESPACE} job/milvus-primer-job -f"
else
    warn "Skipping Milvus priming job."
fi

if [[ "${DO_DOWNLOAD_MODEL}" == "true" ]]; then
    info "Starting Granite model download job..."
    "${PROJECT_ROOT}/scripts/download-model.sh" --cluster --namespace "${NAMESPACE}"
fi

if [[ "${DO_VALIDATE}" == "true" ]]; then
    info "Running deployment validation..."
    "${PROJECT_ROOT}/scripts/validate-deployment.sh" --namespace "${NAMESPACE}"
fi

success "Production deployment finished for namespace '${NAMESPACE}'."
info "Monitor pods with: oc get pods -n ${NAMESPACE} -w"
info "Get UI route with: oc get route streamlit-ui -n ${NAMESPACE}"
