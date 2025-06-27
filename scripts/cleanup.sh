#!/bin/bash

# A script to safely and completely remove all resources created for the
# Meridian Retail AI Demo by deleting the entire OpenShift project.

# Exit immediately if a command exits with a non-zero status.
set -e

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

# --- Main Logic ---
NAMESPACE="retail-ai-demo"

info "This script will permanently delete all resources associated with the demo."
warn "This includes all deployments, services, configmaps, secrets, and the project itself."
echo ""

# --- Safety Confirmation Prompt ---
read -p "Are you sure you want to delete the entire '${NAMESPACE}' project? [y/N] " -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    info "Cleanup aborted by user."
    exit 0
fi

# --- Cleanup Execution ---
info "Proceeding with deletion..."

if ! oc get project ${NAMESPACE} > /dev/null 2>&1; then
    success "Project '${NAMESPACE}' does not exist. Nothing to clean up."
    exit 0
fi

info "Deleting project '${NAMESPACE}'. This may take a few moments..."
if oc delete project ${NAMESPACE}; then
    success "Project '${NAMESPACE}' has been marked for deletion."
    info "You can monitor its termination with: 'oc get project ${NAMESPACE}'"
else
    error "Failed to delete project '${NAMESPACE}'. Please check for errors or permissions issues."
fi

echo ""
success "Cleanup script finished."