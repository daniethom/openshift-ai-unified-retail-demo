#!/bin/bash

# A script to deploy the Meridian Retail AI Demo to a local OpenShift (CRC) cluster.
# This uses the 'local' Kustomize overlay for a resource-constrained environment.

# --- Style Functions ---
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}INFO: $1${NC}"
}

success() {
    echo -e "${GREEN}SUCCESS: $1${NC}"
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
    error "Not logged into an OpenShift cluster. Please run 'oc login...' to connect to your CRC instance."
fi

# --- Deployment ---
info "Starting deployment to OpenShift Local (CRC)..."
NAMESPACE="retail-ai-demo"

info "Applying Kustomize overlay for 'local' environment..."
if oc apply -k ../k8s/overlays/local; then
    success "Deployment manifests applied successfully to namespace '${NAMESPACE}'."
    info "Run 'oc get pods -n ${NAMESPACE} -w' to monitor the pod status."
else
    error "Failed to apply Kustomize configuration. Please check the output for errors."
fi

echo ""
success "Local deployment script finished."

echo "--- Applying Database Priming Job ---"
kubectl apply -f k8s/base/priming-job.yaml