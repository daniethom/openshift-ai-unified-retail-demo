"""Agent status and collaboration overview."""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from streamlit_app.components.agent_status import (
    AgentMetricsTracker,
    AgentStatusDisplay,
    render_agent_comparison_table,
)
from streamlit_app.page_utils import ensure_agent_system, get_agent_statuses

st.title("🤖 AI Agents")
st.caption("Live status for the Meridian multi-agent crew")

if not ensure_agent_system():
    st.error("Unable to initialize the agent system. Check MCP servers and refresh.")
    st.stop()

status_display = AgentStatusDisplay()
metrics_tracker = AgentMetricsTracker()
agent_statuses = get_agent_statuses()

status_display.render_system_metrics(agent_statuses)
st.divider()

tabs = st.tabs(["Agent Grid", "Comparison", "Performance", "Collaboration", "Activity"])

with tabs[0]:
    status_display.render_agent_grid(agent_statuses)

with tabs[1]:
    render_agent_comparison_table(agent_statuses)

with tabs[2]:
    status_display.render_performance_chart(agent_statuses)
    metrics_tracker.record_metrics(agent_statuses)
    metrics_tracker.render_trends()

with tabs[3]:
    status_display.render_collaboration_network(agent_statuses)

with tabs[4]:
    status_display.render_activity_timeline(st.session_state.get("messages", []))
