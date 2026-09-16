"""Shared helpers for Streamlit multipage navigation."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import httpx
import streamlit as st

from config.settings import settings


def ensure_project_root() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_session_defaults() -> None:
    for key, value in {
        "messages": [],
        "agent_system": None,
        "system_initialized": False,
        "current_query_id": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value


async def _initialize_agent_system() -> bool:
    from streamlit_app.app import AgentSystemInterface

    agent_system = AgentSystemInterface()
    success = await agent_system.initialize()
    if success:
        st.session_state.agent_system = agent_system
        st.session_state.system_initialized = True
    return success


def ensure_agent_system() -> bool:
    """Initialize the agent crew if the user landed on a sub-page first."""
    ensure_project_root()
    ensure_session_defaults()
    if st.session_state.system_initialized and st.session_state.agent_system:
        return True

    with st.spinner("Initializing Meridian Retail AI agents..."):
        return asyncio.run(_initialize_agent_system())


def demo_agent_statuses() -> dict[str, dict[str, Any]]:
    """Fallback metrics when the live agent system is unavailable."""
    return {
        "HomeAgent": {
            "status": "active",
            "role": "Chief AI Orchestrator",
            "metrics": {
                "queries_processed": 425,
                "success_rate": 0.98,
                "avg_response_time": 2.1,
                "total_collaborations": 380,
            },
            "capabilities": [
                "Query Analysis",
                "Agent Orchestration",
                "Response Synthesis",
            ],
        },
        "InventoryAgent": {
            "status": "active",
            "role": "Inventory Management Specialist",
            "metrics": {
                "queries_processed": 120,
                "success_rate": 0.96,
                "avg_response_time": 1.8,
                "total_collaborations": 95,
            },
            "capabilities": ["Stock Checking", "Demand Forecasting", "Reorder Planning"],
        },
        "PricingAgent": {
            "status": "active",
            "role": "Revenue Optimization Specialist",
            "metrics": {
                "queries_processed": 95,
                "success_rate": 0.94,
                "avg_response_time": 2.3,
                "total_collaborations": 85,
            },
            "capabilities": [
                "Price Optimization",
                "Competitive Analysis",
                "Promotion Planning",
            ],
        },
        "CustomerAgent": {
            "status": "active",
            "role": "Customer Experience Specialist",
            "metrics": {
                "queries_processed": 85,
                "success_rate": 0.97,
                "avg_response_time": 1.5,
                "total_collaborations": 70,
            },
            "capabilities": [
                "Customer Support",
                "Personalization",
                "Loyalty Management",
            ],
        },
        "TrendAgent": {
            "status": "active",
            "role": "Fashion Trend Analyst",
            "metrics": {
                "queries_processed": 75,
                "success_rate": 0.95,
                "avg_response_time": 2.7,
                "total_collaborations": 65,
            },
            "capabilities": [
                "Trend Analysis",
                "Market Intelligence",
                "Seasonal Forecasting",
            ],
        },
    }


def get_agent_statuses() -> dict[str, dict[str, Any]]:
    ensure_session_defaults()
    if st.session_state.system_initialized and st.session_state.agent_system:
        return st.session_state.agent_system.get_agent_statuses()
    return demo_agent_statuses()


def fetch_mcp_health(url: str) -> dict[str, Any]:
    try:
        response = httpx.get(f"{url.rstrip('/')}/healthz", timeout=5.0)
        response.raise_for_status()
        return {"ok": True, "status_code": response.status_code, "body": response.json()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def invoke_mcp(url: str, payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    try:
        response = httpx.post(
            f"{url.rstrip('/')}/invoke",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return {"ok": True, "status_code": response.status_code, "body": response.json()}
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        return {
            "ok": False,
            "error": detail or str(exc),
            "status_code": exc.response.status_code,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def mcp_server_rows() -> list[dict[str, str]]:
    return [
        {"name": "LLM", "url": settings.llm_mcp_url, "kind": "llm"},
        {"name": "RAG", "url": settings.rag_mcp_url, "kind": "rag"},
        {"name": "Search", "url": settings.search_mcp_url, "kind": "search"},
        {"name": "Analytics", "url": settings.analytics_mcp_url, "kind": "analytics"},
    ]
