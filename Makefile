# Makefile for Meridian Retail AI Demo

# --- Variables ---
PYTHON = python
VENV_DIR = .venv
SHELL := /bin/bash

.DEFAULT_GOAL := help

# --- Targets ---

help: ## ✨ Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## 📦 Install all project dependencies into a virtual environment
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
		echo "Virtual environment created at $(VENV_DIR)"; \
	fi
	@source $(VENV_DIR)/bin/activate; \
	pip install uv; \
	uv pip install --system -e ".[dev]";
	@echo "✅ Dependencies installed successfully."

.PHONY: lint
lint: ## 🔎 Lint the code using Ruff and check formatting with Black
	@echo "Running Ruff linter..."
	@ruff check .
	@echo "Checking formatting with Black..."
	@black --check .
	@echo "✅ Lint checks passed."

.PHONY: format
format: ## 🎨 Automatically format the code with Ruff and Black
	@echo "Formatting with Ruff..."
	@ruff check . --fix
	@echo "Formatting with Black..."
	@black .
	@echo "✅ Code formatted successfully."

.PHONY: test
test: ## 🧪 Run the full Pytest test suite
	@echo "Running Pytest test suite..."
	@pytest
	@echo "✅ Tests completed."

.PHONY: build-images
build-images: ## 🐳 Build container images (placeholder)
	@echo "Building container images..."
	@echo "--> docker build -t meridian-retail-ai-demo:latest ."
	@echo "✅ Placeholder for image build complete."

.PHONY: run-ui
run-ui: ## 🚀 Run the Streamlit UI locally
	@echo "Starting the Streamlit application..."
	@streamlit run streamlit_app/app.py

.PHONY: deploy-local
deploy-local: ## ⚙️ Deploy the application to a local OpenShift (CRC) cluster
	@echo "Deploying to local cluster..."
	@./scripts/deploy-local.sh

.PHONY: deploy-prod
deploy-prod: ## 🚀 Deploy the application to the production OpenShift AI cluster
	@echo "Deploying to production cluster..."
	@./scripts/deploy-openshift.sh

.PHONY: clean
clean: ## 🧹 Clean up temporary files and directories
	@echo "Cleaning up..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf .pytest_cache
	@rm -rf .ruff_cache
	@rm -rf .mypy_cache
	@echo "✅ Cleanup complete."
