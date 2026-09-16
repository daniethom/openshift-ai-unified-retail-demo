# Stage 1: Builder - Install dependencies into a virtual environment
# Using a slim-bullseye image for a smaller footprint
FROM python:3.11-slim-bullseye AS builder

# Set environment variables to prevent generating .pyc files and to use uv
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install uv, the fast Python package manager
RUN pip install uv

# Create a virtual environment in a standard location
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy package metadata and source needed for editable install
WORKDIR /app
COPY pyproject.toml README.md ./
COPY agents/ agents/
COPY config/ config/
COPY db/ db/
COPY mcp_servers/ mcp_servers/
COPY streamlit_app/ streamlit_app/
COPY rag/ rag/
COPY scripts/ scripts/
COPY alembic/ alembic/
COPY alembic.ini .

# Install into /opt/venv (not system Python) so dependencies are copied to the final image
RUN uv pip install --python /opt/venv/bin/python --no-cache -e ".[model]"


# Stage 2: Final - Create the final, optimized image
# Start from the same base image for consistency
FROM python:3.11-slim-bullseye

# Set the working directory
WORKDIR /app

# Create a non-root user and group for security
RUN addgroup --system appgroup && adduser --system --group appuser

# Copy the virtual environment with pre-installed dependencies from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application source code into the image
COPY . .

# Ensure the app directory is owned by the non-root user
RUN chown -R appuser:appgroup /app

# Switch to the non-root user
USER appuser

# Make the virtual environment's python the default
ENV PATH="/opt/venv/bin:$PATH"

# No CMD or ENTRYPOINT is set.
# The command to run (e.g., uvicorn for a server or streamlit for the UI)
# will be specified in the Kubernetes/OpenShift Deployment manifests.
