#!/bin/bash
# Download Granite model weights locally or trigger the OpenShift download job.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

PROJECT_ROOT="$(script_root)"
NAMESPACE="${OPENSHIFT_NAMESPACE:-retail-ai-demo}"
USE_CLUSTER_JOB=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --cluster           Run the model download Job on OpenShift instead of locally
  --namespace NAME    OpenShift namespace (default: retail-ai-demo)
  -h, --help          Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster) USE_CLUSTER_JOB=true; shift ;;
        --namespace) NAMESPACE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) error "Unknown option: $1" ;;
    esac
done

if [[ "${USE_CLUSTER_JOB}" == "true" ]]; then
    ensure_openshift_login
    info "Applying model download job in namespace ${NAMESPACE}..."
    oc delete job granite-model-download -n "${NAMESPACE}" --ignore-not-found
    oc apply -f "${PROJECT_ROOT}/k8s/overlays/production/model-download-job.yaml"
    info "Watch progress: oc logs -n ${NAMESPACE} job/granite-model-download -f"
    oc wait --for=condition=complete job/granite-model-download -n "${NAMESPACE}" --timeout=3600s
    success "Model download job completed."
    exit 0
fi

require_command python3
info "Downloading model locally..."
python3 "${PROJECT_ROOT}/scripts/download_model.py" "$@"
success "Local model download finished."
