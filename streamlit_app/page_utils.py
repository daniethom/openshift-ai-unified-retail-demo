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


def load_public_config() -> dict[str, str]:
    """Load redacted public config from the current environment."""
    from config.settings import get_public_config

    return get_public_config()


def _format_money(amount: Any) -> str:
    try:
        return f"R{int(amount):,}"
    except (TypeError, ValueError):
        return str(amount)


def _format_product_item(item: dict[str, Any]) -> str:
    name = item.get("name") or item.get("product") or item.get("product_id") or "Item"
    line = str(name)
    if item.get("price") is not None:
        line += f" ({_format_money(item['price'])})"
    reason = item.get("reason")
    if reason:
        line += f" — {reason}"
    complements = item.get("complements") or []
    for complement in complements:
        if isinstance(complement, dict) and complement.get("name"):
            complement_line = complement["name"]
            if complement.get("price") is not None:
                complement_line += f" ({_format_money(complement['price'])})"
            line += f"; bundle with {complement_line}"
    return line


def _format_recommendations_list(items: list[Any]) -> list[str]:
    lines: list[str] = []
    for item in items[:5]:
        if isinstance(item, dict):
            if item.get("name") or item.get("product_id"):
                lines.append(_format_product_item(item))
            elif item.get("message"):
                lines.append(str(item["message"]))
            elif item.get("action"):
                detail = item.get("description") or item.get("trend") or ""
                lines.append(f"{item['action']}: {detail}".strip(": "))
            elif item.get("type") and item.get("trend"):
                action = str(item.get("action", "introduce")).title()
                category = str(item.get("category", "category")).replace("_", " ")
                priority = item.get("priority", "medium")
                lines.append(
                    f"{action} {category} for {item['trend']} ({priority} priority)"
                )
        elif isinstance(item, str) and item.strip():
            lines.append(item.strip())
    return lines


def _format_specialist_result(result: dict[str, Any]) -> str | None:
    sections: list[str] = []

    recommendations = result.get("recommendations")
    if isinstance(recommendations, list):
        if (
            recommendations
            and isinstance(recommendations[0], dict)
            and (recommendations[0].get("name") or recommendations[0].get("product_id"))
        ):
            product_lines = _format_recommendations_list(recommendations)
            if product_lines:
                sections.append(
                    "Recommended products:\n"
                    + "\n".join(f"  • {line}" for line in product_lines)
                )
        else:
            text_lines = _format_recommendations_list(recommendations)
            if text_lines:
                sections.append("Recommendations: " + "; ".join(text_lines))

    cross_sell = result.get("cross_sell_opportunities")
    if isinstance(cross_sell, list):
        cross_lines = [
            item.get("message")
            for item in cross_sell
            if isinstance(item, dict) and item.get("message")
        ]
        if cross_lines:
            sections.append("Cross-sell: " + "; ".join(cross_lines[:3]))

    if result.get("primary_insight"):
        sections.append(str(result["primary_insight"]))

    optimizations = result.get("optimizations")
    if isinstance(optimizations, list) and optimizations:
        optimization_lines = []
        for item in optimizations[:4]:
            if isinstance(item, dict) and item.get("action"):
                line = str(item["action"])
                if item.get("impact"):
                    line += f" ({item['impact']} impact)"
                if item.get("stores"):
                    line += f" — {item['stores']}"
                optimization_lines.append(line)
        if optimization_lines:
            sections.append(
                "Actions:\n" + "\n".join(f"  • {line}" for line in optimization_lines)
            )

    potential_savings = result.get("potential_savings")
    if isinstance(potential_savings, dict) and potential_savings.get("annual_savings"):
        sections.append(
            "Potential savings: "
            f"{_format_money(potential_savings['annual_savings'])} annually"
        )

    implementation_plan = result.get("implementation_plan")
    if isinstance(implementation_plan, list) and implementation_plan:
        plan_lines = []
        for step in implementation_plan[:3]:
            if isinstance(step, dict) and step.get("action"):
                timeline = step.get("timeline", "")
                prefix = f"Step {step['step']}: " if step.get("step") else ""
                suffix = f" ({timeline})" if timeline else ""
                plan_lines.append(f"{prefix}{step['action']}{suffix}")
        if plan_lines:
            sections.append("Plan:\n" + "\n".join(f"  • {line}" for line in plan_lines))

    resolution = result.get("resolution")
    if isinstance(resolution, dict):
        if resolution.get("immediate_action"):
            sections.append(f"Immediate action: {resolution['immediate_action']}")
        if resolution.get("apology"):
            sections.append(f"Apology: {resolution['apology']}")
        if resolution.get("long_term_solution"):
            sections.append(f"Long-term fix: {resolution['long_term_solution']}")

    compensation = result.get("compensation")
    if isinstance(compensation, dict):
        compensation_line = (
            f"Compensation: {str(compensation.get('type', 'offer')).replace('_', ' ')}"
        )
        if compensation.get("amount") is not None:
            compensation_line += f" ({_format_money(compensation['amount'])})"
        if compensation.get("additional"):
            compensation_line += f" + {compensation['additional']}"
        sections.append(compensation_line)

    follow_up_plan = result.get("follow_up_plan")
    if isinstance(follow_up_plan, list) and follow_up_plan:
        follow_up_lines = []
        for step in follow_up_plan[:3]:
            if isinstance(step, dict) and step.get("action"):
                timing = step.get("timing", "")
                prefix = f"{timing}: " if timing else ""
                follow_up_lines.append(f"{prefix}{step['action']}")
        if follow_up_lines:
            sections.append("Follow-up: " + "; ".join(follow_up_lines))

    availability = result.get("complement_availability")
    if isinstance(availability, list):
        availability_lines = []
        for row in availability[:4]:
            if isinstance(row, dict):
                product = row.get("product", "Item")
                location = row.get("location", "")
                units = row.get("available")
                suffix = f" ({units} units)" if units is not None else ""
                location_suffix = f" @ {location}" if location else ""
                availability_lines.append(f"{product}{location_suffix}{suffix}")
        if availability_lines:
            sections.append("Availability: " + "; ".join(availability_lines))

    bundle_offers = result.get("bundle_offers")
    if isinstance(bundle_offers, list):
        offer_lines = [
            offer.get("offer") or offer.get("description")
            for offer in bundle_offers
            if isinstance(offer, dict)
            and (offer.get("offer") or offer.get("description"))
        ]
        if offer_lines:
            sections.append("Offers: " + "; ".join(offer_lines))

    alerts = result.get("alerts")
    if isinstance(alerts, list) and not result.get("primary_insight"):
        alert_lines = [
            alert.get("message")
            for alert in alerts[:2]
            if isinstance(alert, dict) and alert.get("message")
        ]
        if alert_lines:
            sections.append("Alerts: " + "; ".join(alert_lines))

    reasoning = result.get("reasoning")
    if reasoning:
        if sections:
            sections.append(f"Why: {reasoning}")
        else:
            sections.append(str(reasoning))

    if sections:
        return "\n".join(sections)

    text = _format_insight_value(result)
    return text or None


def _format_insight_value(value: Any, *, depth: int = 0) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for key in ("primary_insight", "summary", "response", "message"):
            nested = value.get(key)
            if nested:
                return _format_insight_value(nested, depth=depth + 1)
        key_trends = value.get("key_trends") or value.get("top_trends")
        if isinstance(key_trends, list):
            names = [
                trend.get("name")
                for trend in key_trends
                if isinstance(trend, dict) and trend.get("name")
            ]
            if names:
                return "Key trends: " + ", ".join(names[:5])
        recommendations = value.get("recommendations")
        if isinstance(recommendations, list) and recommendations:
            if (
                depth == 0
                and isinstance(recommendations[0], dict)
                and (
                    recommendations[0].get("name")
                    or recommendations[0].get("product_id")
                )
            ):
                lines = _format_recommendations_list(recommendations)
                if lines:
                    return "; ".join(lines)
            text = _format_insight_value(recommendations[0], depth=depth + 1)
            if text:
                return text
        if depth == 0:
            parts = []
            for key, nested in list(value.items())[:3]:
                text = _format_insight_value(nested, depth=depth + 1)
                if text:
                    parts.append(f"{key.replace('_', ' ').title()}: {text}")
            if parts:
                return "; ".join(parts)
        return ""
    if isinstance(value, list):
        parts = [_format_insight_value(item, depth=depth + 1) for item in value[:3]]
        return "; ".join(part for part in parts if part)
    return str(value).strip()


def format_agent_insight(insight: Any) -> str | None:
    """Extract a concise insight line from a specialist agent payload."""
    if insight is None:
        return None
    if not isinstance(insight, dict):
        text = _format_insight_value(insight)
        return text or None

    if insight.get("status") == "error":
        return f"Unable to complete analysis ({insight.get('error', 'unknown error')})"

    sections: list[str] = []

    result_payload = insight.get("result")
    if isinstance(result_payload, dict):
        text = _format_specialist_result(result_payload)
        if text:
            sections.append(text)

    if insight.get("analysis"):
        analysis = insight["analysis"]
        if isinstance(analysis, dict):
            text = _format_insight_value(analysis)
            if text:
                sections.append(text)
            regional = analysis.get("regional_insights")
            if regional:
                sections.append(str(regional))

    if insight.get("recommendations"):
        rec_lines = _format_recommendations_list(insight["recommendations"])
        if rec_lines:
            sections.append("Actions: " + "; ".join(rec_lines))

    if sections:
        return "\n".join(sections)

    for key in ("summary", "primary_insight"):
        if key in insight and insight[key] is not None:
            text = _format_insight_value(insight[key])
            if text:
                return text

    if insight.get("recommendations"):
        text = _format_insight_value(insight["recommendations"])
        if text:
            return text

    if insight.get("status") == "success":
        payload = {
            key: value
            for key, value in insight.items()
            if key not in {"metadata", "query", "status", "context"}
        }
        text = _format_insight_value(payload)
        return text or None

    return None


def format_assistant_response(response: dict[str, Any]) -> str:
    """Format the orchestrator response for chat display."""
    summary = response.get("summary", "I've analyzed your query.")
    formatted = summary

    insights = response.get("detailed_insights", {})
    insight_lines: list[str] = []
    if isinstance(insights, dict):
        for agent, insight in insights.items():
            text = format_agent_insight(insight)
            if text:
                if "\n" in text:
                    insight_lines.append(f"- **{agent}**:\n{text}")
                else:
                    insight_lines.append(f"- **{agent}**: {text}")

    if insight_lines:
        formatted += "\n\n**Key Insights:**\n" + "\n".join(insight_lines)

    recommendations = response.get("recommendations", [])
    if recommendations:
        formatted += "\n\n**Recommendations:**\n"
        for index, rec in enumerate(recommendations, 1):
            if isinstance(rec, dict):
                rec_text = rec.get("action", rec.get("description", str(rec)))
            else:
                rec_text = str(rec)
            formatted += f"{index}. {rec_text}\n"

    return formatted.strip()


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
            "capabilities": [
                "Stock Checking",
                "Demand Forecasting",
                "Reorder Planning",
            ],
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
        return {
            "ok": True,
            "status_code": response.status_code,
            "body": response.json(),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def invoke_mcp(
    url: str, payload: dict[str, Any], timeout: float = 60.0
) -> dict[str, Any]:
    try:
        response = httpx.post(
            f"{url.rstrip('/')}/invoke",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return {
            "ok": True,
            "status_code": response.status_code,
            "body": response.json(),
        }
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
