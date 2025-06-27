#!/bin/bash

# A script to run different suites of tests for the Meridian Retail AI Demo.
# It can run all tests, or be limited to 'unit', 'integration', or 'e2e' tests.

# Exit immediately if a command exits with a non-zero status.
set -e

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

usage() {
    echo "Usage: $0 [unit|integration|e2e]"
    echo "  (no argument): Runs all tests."
    echo "  unit:          Runs only unit tests."
    echo "  integration:   Runs only integration tests."
    echo "  e2e:           Runs only end-to-end tests."
    exit 1
}

# --- Main Logic ---
TEST_PATH="tests/"
TEST_TYPE="all"

# Check for command-line argument
if [ "$#" -gt 1 ]; then
    error "Too many arguments."
    usage
elif [ "$#" -eq 1 ]; then
    case "$1" in
        unit)
            TEST_PATH="tests/unit/"
            TEST_TYPE="unit"
            ;;
        integration)
            TEST_PATH="tests/integration/"
            TEST_TYPE="integration"
            ;;
        e2e)
            TEST_PATH="tests/e2e/"
            TEST_TYPE="end-to-end"
            ;;
        *)
            error "Unknown test type '$1'."
            usage
            ;;
    esac
fi

info "Running ${TEST_TYPE} tests from path: ${TEST_PATH}"

# Execute pytest with verbose output and code coverage analysis
# The --cov=. flag tells pytest-cov to analyze coverage for the entire project.
pytest -v --cov=. "${TEST_PATH}"

success "All ${TEST_TYPE} tests passed successfully."
