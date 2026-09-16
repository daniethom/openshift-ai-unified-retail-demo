# Makefile for Meridian Retail AI Demo

# --- Variables ---
PYTHON = python3
VENV_DIR = .venv
NAMESPACE = retail-ai-demo
SHELL := /bin/bash

.DEFAULT_GOAL := help

# --- Targets ---

help: ## ✨ Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## 📦 Install all project dependencies into a virtual environment
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
		echo "Virtual environment created at $(VENV_DIR)"; \
	fi
	@source $(VENV_DIR)/bin/activate; \
	pip install uv; \
	uv pip install --system -e ".[dev,model]";
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
	@RAG_USE_FALLBACK=true pytest
	@echo "✅ Tests completed."

.PHONY: build-images
build-images: ## 🐳 Build and push image to OpenShift internal registry
	@./scripts/build-image.sh --namespace $(NAMESPACE)

.PHONY: build-images-local
build-images-local: ## 🐳 Build the container image locally without pushing
	@./scripts/build-image.sh --local-only

.PHONY: build-images-openshift
build-images-openshift: ## 🏗️ Build the image on-cluster with OpenShift BuildConfig
	@./scripts/build-image.sh --openshift-build --namespace $(NAMESPACE)

.PHONY: download-model
download-model: ## ⬇️ Download Granite model weights locally
	@./scripts/download-model.sh

.PHONY: download-model-cluster
download-model-cluster: ## ⬇️ Download Granite model weights onto the cluster PVC
	@./scripts/download-model.sh --cluster --namespace $(NAMESPACE)

.PHONY: validate
validate: ## ✅ Run pre-demo deployment validation checks
	@./scripts/validate-deployment.sh --namespace $(NAMESPACE)

.PHONY: validate-local
validate-local: ## ✅ Validate a local/CRC deployment (skip kServe checks)
	@./scripts/validate-deployment.sh --namespace $(NAMESPACE) --skip-inference

.PHONY: demo-checklist
demo-checklist: ## 📋 Print demo-day routes, status, and sample queries
	@./scripts/demo-day-checklist.sh --namespace $(NAMESPACE)

.PHONY: demo-checklist-local
demo-checklist-local: ## 📋 Demo-day checklist for local/CRC (skip kServe)
	@./scripts/demo-day-checklist.sh --namespace $(NAMESPACE) --skip-inference

.PHONY: demo-checklist-strict
demo-checklist-strict: ## 📋 Demo-day checklist; fail if anything is not ready
	@./scripts/demo-day-checklist.sh --namespace $(NAMESPACE) --strict --validate

.PHONY: run-ui
run-ui: ## 🚀 Run the Streamlit UI locally
	@echo "Starting the Streamlit application..."
	@streamlit run streamlit_app/app.py

.PHONY: deploy-local
deploy-local: ## ⚙️ Deploy to local OpenShift (CRC)
	@./scripts/deploy-local.sh

.PHONY: deploy-local-full
deploy-local-full: ## ⚙️ Build, deploy, prime, and validate on local OpenShift
	@./scripts/deploy-local.sh --build --validate

.PHONY: deploy-prod
deploy-prod: ## 🚀 Deploy to production OpenShift AI cluster
	@./scripts/deploy-openshift.sh

.PHONY: deploy-prod-full
deploy-prod-full: ## 🚀 Build, deploy, download model, prime, and validate production
	@./scripts/deploy-openshift.sh --build --download-model --validate

.PHONY: clean
clean: ## 🧹 Clean up temporary files and directories
	@echo "Cleaning up..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf .pytest_cache
	@rm -rf .ruff_cache
	@rm -rf .mypy_cache
	@echo "✅ Cleanup complete."
