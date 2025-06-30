#!/bin/bash

# A script to deploy the Meridian Retail AI Demo to a production OpenShift AI cluster.
# This uses the 'production' Kustomize overlay.

# --- Style Functions ---
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}INFO: $1${NC}"
}

success() {
    echo -e "${GREEN}SUCCESS: $1${NC}"
}

warn() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit 1
}

# --- Pre-flight Checks ---
info "Checking for required tools..."

if ! command -v oc &> /dev/null; then
    error "'oc' command not found. Please ensure the OpenShift CLI is installed and in your PATH."
fi

info "Checking OpenShift login status..."
if ! oc whoami &> /dev/null; then
    warn "You are not logged into an OpenShift cluster."
    warn "Please ensure you log in to your PRODUCTION cluster before proceeding."
    read -p "Press [Enter] to continue if you are sure, or Ctrl+C to abort."
fi

# --- Deployment ---
info "Starting deployment to PRODUCTION OpenShift AI cluster..."
NAMESPACE="retail-ai-demo"

# You would ideally have separate logic here to build and push container images
# to the OpenShift internal registry before applying manifests.
# For now, we assume images are already available.
info "Building and pushing images is currently skipped. Assuming images exist in the cluster."

info "Applying Kustomize overlay for 'production' environment..."
if oc apply -k ../k8s/overlays/production; then
    success "Deployment manifests applied successfully to namespace '${NAMESPACE}'."
    info "Run 'oc get pods -n ${NAMESPACE} -w' to monitor the pod status."
else
    error "Failed to apply Kustomize configuration. Please check the output for errors."
fi

echo ""
success "Production deployment script finished."

echo "--- Applying Database Priming Job ---"
kubectl apply -f k8s/base/priming-job.yaml