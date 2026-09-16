#!/bin/bash
# Build and push the application image to the OpenShift internal registry.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

PROJECT_ROOT="$(script_root)"
NAMESPACE="${OPENSHIFT_NAMESPACE:-retail-ai-demo}"
IMAGE_NAME="${IMAGE_NAME:-meridian-retail-ai}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
USE_OPENSHIFT_BUILD=false
LOCAL_ONLY=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --openshift-build   Use 'oc start-build' instead of local podman/docker
  --local-only        Build locally without pushing to the registry
  --namespace NAME    OpenShift namespace (default: retail-ai-demo)
  --tag TAG           Image tag (default: latest)
  -h, --help          Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --openshift-build) USE_OPENSHIFT_BUILD=true; shift ;;
        --local-only) LOCAL_ONLY=true; shift ;;
        --namespace) NAMESPACE="$2"; shift 2 ;;
        --tag) IMAGE_TAG="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) error "Unknown option: $1" ;;
    esac
done

if [[ "${USE_OPENSHIFT_BUILD}" == "true" ]]; then
    ensure_openshift_login
    ensure_namespace "${NAMESPACE}"
    info "Starting OpenShift binary build for ${IMAGE_NAME}:${IMAGE_TAG}..."
    oc apply -f "${PROJECT_ROOT}/k8s/overlays/production/imagestream.yaml" >/dev/null 2>&1 || true
    oc apply -f "${PROJECT_ROOT}/k8s/overlays/production/buildconfig.yaml"
    oc start-build "${IMAGE_NAME}" --from-dir="${PROJECT_ROOT}" --follow -n "${NAMESPACE}"
    success "OpenShift build completed: $(openshift_image_reference "${NAMESPACE}" "${IMAGE_NAME}" "${IMAGE_TAG}")"
    exit 0
fi

BUILDER="$(detect_container_builder)"
IMAGE_REF="$(openshift_image_reference "${NAMESPACE}" "${IMAGE_NAME}" "${IMAGE_TAG}")"

info "Building image with ${BUILDER}: ${IMAGE_REF}"
"${BUILDER}" build -t "${IMAGE_REF}" "${PROJECT_ROOT}"

if [[ "${LOCAL_ONLY}" == "true" ]]; then
    success "Local image built: ${IMAGE_REF}"
    exit 0
fi

ensure_openshift_login
ensure_namespace "${NAMESPACE}"
info "Logging into OpenShift internal registry..."
oc registry login
info "Pushing image..."
"${BUILDER}" push "${IMAGE_REF}"
success "Image pushed: ${IMAGE_REF}"
