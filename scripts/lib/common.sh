#!/bin/bash
# Shared helpers for deployment and validation scripts.

BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

info() { echo -e "${BLUE}INFO: $1${NC}"; }
success() { echo -e "${GREEN}SUCCESS: $1${NC}"; }
warn() { echo -e "${YELLOW}WARNING: $1${NC}"; }
error() { echo -e "${RED}ERROR: $1${NC}" >&2; exit 1; }

script_root() {
    local source="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
    cd "$(dirname "${source}")/.." && pwd
}

require_command() {
    local cmd="$1"
    command -v "${cmd}" >/dev/null 2>&1 || error "'${cmd}' is required but not installed."
}

detect_container_builder() {
    if command -v podman >/dev/null 2>&1; then
        echo "podman"
    elif command -v docker >/dev/null 2>&1; then
        echo "docker"
    else
        error "Install podman or docker to build container images."
    fi
}

openshift_image_reference() {
    local namespace="${1:-retail-ai-demo}"
    local image_name="${2:-meridian-retail-ai}"
    local tag="${3:-latest}"
    echo "image-registry.openshift-image-registry.svc:5000/${namespace}/${image_name}:${tag}"
}

ensure_openshift_login() {
    require_command oc
    oc whoami >/dev/null 2>&1 || error "Not logged into OpenShift. Run 'oc login' first."
}

ensure_namespace() {
    local namespace="${1:-retail-ai-demo}"
    if ! oc get namespace "${namespace}" >/dev/null 2>&1; then
        info "Creating namespace ${namespace}..."
        oc create namespace "${namespace}"
    fi
}
