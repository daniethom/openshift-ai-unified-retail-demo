#!/bin/bash

# A script to set up the development environment for the Meridian Retail AI Demo.
# It checks for dependencies and guides the user through local or OpenShift setup.

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

# --- Prerequisite Checks ---
info "Step 1: Checking for required tools..."
command -v python3 >/dev/null 2>&1 || error "Python 3 is not installed. Please install it to continue."
command -v uv >/dev/null 2>&1 || command -v pip >/dev/null 2>&1 || error "Neither 'uv' nor 'pip' is installed. Please install one to manage Python packages."
command -v podman >/dev/null 2>&1 || error "Podman is not installed. Please install it to run local services."
command -v oc >/dev/null 2>&1 || warn "'oc' (OpenShift CLI) is not installed. You will not be able to perform OpenShift-related tasks."
success "All primary tools are available."
echo ""

# --- Interactive Setup Choice ---
info "Step 2: Choose your setup scenario."
PS3="Please enter your choice: "
options=("Local Development Setup" "OpenShift Deployment Preparation" "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "Local Development Setup")
            info "Starting Local Development Setup..."

            # Check if Podman service is running
            if ! podman info >/dev/null 2>&1; then
                error "Podman service is not running. Please start Podman and run this script again."
            fi
            success "Podman service is running."

            # Check for .env file
            if [ ! -f ../.env ]; then
                info "No .env file found. Copying from .env.example..."
                cp ../.env.example ../.env
                success ".env file created."
                warn "Please edit the .env file now and add your TAVILY_API_KEY."
            else
                success ".env file already exists."
            fi

            # Set up Python virtual environment
            if [ ! -d ../.venv ]; then
                info "Creating Python virtual environment at ../.venv..."
                python3 -m venv ../.venv
            fi
            info "Activating virtual environment and installing dependencies..."
            # Activate and install using uv if available, otherwise pip
            if command -v uv >/dev/null 2>&1; then
                ../.venv/bin/uv pip install --system -e "..[dev]"
            else
                ../.venv/bin/pip install -e "..[dev]"
            fi
            success "Python environment is set up and dependencies are installed."
            
            info "Next steps for local development:"
            echo "1. Run 'make test' to verify the setup."
            echo "2. Run 'make run-ui' to start the application."
            break
            ;;
        "OpenShift Deployment Preparation")
            if ! command -v oc >/dev/null 2>&1; then
                error "'oc' command not found. Cannot proceed with OpenShift setup."
            fi
            info "Starting OpenShift Deployment Preparation..."
            
            # Verify login
            if ! oc whoami &> /dev/null; then
                error "Not logged into an OpenShift cluster. Please run 'oc login...' first."
            fi
            success "Logged into OpenShift as '$(oc whoami)'."
            
            # Check for project/namespace
            NAMESPACE="retail-ai-demo"
            if ! oc get project ${NAMESPACE} > /dev/null 2>&1; then
                info "Project '${NAMESPACE}' not found. Creating it now..."
                oc new-project ${NAMESPACE}
                success "Project '${NAMESPACE}' created."
            else
                success "Project '${NAMESPACE}' already exists."
            fi
            
            # Remind about secret creation
            warn "Reminder: Ensure you have created the Kubernetes secret for your API key."
            echo "You can create it with a command like this:"
            echo "oc create secret generic mcp-servers-secrets --from-literal=TAVILY_API_KEY='your_key_here' -n ${NAMESPACE}"
            
            info "Next steps for OpenShift deployment:"
            echo "1. Build and push your container images."
            echo "2. Run 'make deploy-local' or 'make deploy-prod' to deploy the application."
            break
            ;;
        "Quit")
            break
            ;;
        *) echo "Invalid option $REPLY";;
    esac
done

echo ""
success "Setup script finished."
