"""
Agent Status Component for Meridian Retail AI
Displays real-time agent status and metrics
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


class AgentStatusDisplay:
    """
    Manages the display of agent status and metrics
    """
    
    def __init__(self):
        self.status_colors = {
            "active": "#4CAF50",
            "busy": "#FF9800",
            "error": "#F44336",
            "offline": "#9E9E9E"
        }
        
        self.status_icons = {
            "active": "✅",
            "busy": "⏳",
            "error": "❌",
            "offline": "⭕"
        }
    
    def render_agent_card(self, agent_name: str, status: Dict[str, Any]):
        """
        Render a single agent status card
        
        Args:
            agent_name: Name of the agent
            status: Agent status dictionary
        """
        # Get status info
        current_status = status.get("status", "offline")
        status_color = self.status_colors.get(current_status, "#9E9E9E")
        status_icon = self.status_icons.get(current_status, "⭕")
        
        # Create card
        with st.container():
            st.markdown(f"""
            <div style="
                background-color: white;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-left: 4px solid {status_color};
                margin-bottom: 1rem;
            ">
                <h4 style="margin: 0 0 0.5rem 0;">
                    {status_icon} {agent_name}
                </h4>
                <p style="color: #666; margin: 0;">
                    <strong>Role:</strong> {status.get('role', 'Unknown')}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics in columns
            col1, col2, col3 = st.columns(3)
            
            metrics = status.get("metrics", {})
            
            with col1:
                st.metric(
                    "Queries",
                    metrics.get("queries_processed", 0),
                    delta=None
                )
            
            with col2:
                success_rate = metrics.get("success_rate", 1.0)
                st.metric(
                    "Success Rate",
                    f"{success_rate * 100:.1f}%",
                    delta=None
                )
            
            with col3:
                avg_time = metrics.get("avg_response_time", 0)
                st.metric(
                    "Avg Time",
                    f"{avg_time:.2f}s",
                    delta=None
                )
            
            # Current task (if busy)
            if current_status == "busy" and status.get("current_task"):
                st.info(f"🔄 {status['current_task']}")
            
            # Capabilities
            capabilities = status.get("capabilities", [])
            if capabilities:
                with st.expander("View Capabilities"):
                    for cap in capabilities:
                        st.write(f"• {cap}")
    
    def render_agent_grid(self, agent_statuses: Dict[str, Dict[str, Any]]):
        """
        Render all agents in a grid layout
        
        Args:
            agent_statuses: Dictionary of agent statuses
        """
        # Home agent separately (as orchestrator)
        if "HomeAgent" in agent_statuses:
            st.markdown("### 🎯 Orchestrator")
            self.render_agent_card("HomeAgent", agent_statuses["HomeAgent"])
            st.divider()
        
        # Specialist agents
        st.markdown("### 🤖 Specialist Agents")
        
        # Create two columns for specialist agents
        specialist_agents = {k: v for k, v in agent_statuses.items() if k != "HomeAgent"}
        
        cols = st.columns(2)
        for i, (agent_name, status) in enumerate(specialist_agents.items()):
            with cols[i % 2]:
                self.render_agent_card(agent_name, status)
    
    def render_system_metrics(self, agent_statuses: Dict[str, Dict[str, Any]]):
        """
        Render overall system metrics
        
        Args:
            agent_statuses: Dictionary of agent statuses
        """
        st.markdown("### 📊 System Metrics")
        
        # Calculate aggregate metrics
        total_queries = sum(
            status.get("metrics", {}).get("queries_processed", 0)
            for status in agent_statuses.values()
        )
        
        active_agents = sum(
            1 for status in agent_statuses.values()
            if status.get("status") == "active"
        )
        
        avg_success = sum(
            status.get("metrics", {}).get("success_rate", 1.0)
            for status in agent_statuses.values()
        ) / len(agent_statuses) if agent_statuses else 0
        
        total_collaborations = sum(
            status.get("metrics", {}).get("total_collaborations", 0)
            for status in agent_statuses.values()
        )
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Queries",
                total_queries,
                delta="+5" if total_queries > 0 else None
            )
        
        with col2:
            st.metric(
                "Active Agents",
                f"{active_agents}/{len(agent_statuses)}",
                delta=None
            )
        
        with col3:
            st.metric(
                "Success Rate",
                f"{avg_success * 100:.1f}%",
                delta="+2.3%" if avg_success > 0.9 else None
            )
        
        with col4:
            st.metric(
                "Collaborations",
                total_collaborations,
                delta=None
            )
    
    def render_performance_chart(self, agent_statuses: Dict[str, Dict[str, Any]]):
        """
        Render agent performance comparison chart
        
        Args:
            agent_statuses: Dictionary of agent statuses
        """
        st.markdown("### 📈 Agent Performance")
        
        # Prepare data for visualization
        data = []
        for agent_name, status in agent_statuses.items():
            metrics = status.get("metrics", {})
            data.append({
                "Agent": agent_name,
                "Queries": metrics.get("queries_processed", 0),
                "Success Rate": metrics.get("success_rate", 1.0) * 100,
                "Avg Response Time": metrics.get("avg_response_time", 0)
            })
        
        df = pd.DataFrame(data)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Query Volume", "Success Rates", "Response Times"])
        
        with tab1:
            fig = px.bar(
                df,
                x="Agent",
                y="Queries",
                title="Queries Processed by Agent",
                color="Queries",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.bar(
                df,
                x="Agent",
                y="Success Rate",
                title="Success Rate by Agent (%)",
                color="Success Rate",
                color_continuous_scale="Greens",
                range_y=[0, 100]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = px.bar(
                df,
                x="Agent",
                y="Avg Response Time",
                title="Average Response Time by Agent (seconds)",
                color="Avg Response Time",
                color_continuous_scale="Oranges"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_collaboration_network(self, agent_statuses: Dict[str, Dict[str, Any]]):
        """
        Render agent collaboration network
        
        Args:
            agent_statuses: Dictionary of agent statuses
        """
        st.markdown("### 🔗 Agent Collaboration Network")
        
        # Create network visualization using plotly
        # Define node positions (simplified circular layout)
        import math
        
        agents = list(agent_statuses.keys())
        n = len(agents)
        
        # Calculate positions
        pos = {}
        for i, agent in enumerate(agents):
            angle = 2 * math.pi * i / n
            if agent == "HomeAgent":
                # Center position for orchestrator
                pos[agent] = (0, 0)
            else:
                # Circular positions for others
                pos[agent] = (math.cos(angle) * 2, math.sin(angle) * 2)
        
        # Create edges (HomeAgent connected to all others)
        edge_trace = []
        for agent in agents:
            if agent != "HomeAgent":
                x0, y0 = pos["HomeAgent"]
                x1, y1 = pos[agent]
                edge_trace.append(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(width=2, color='#888'),
                        hoverinfo='none',
                        showlegend=False
                    )
                )
        
        # Create nodes
        node_trace = go.Scatter(
            x=[pos[agent][0] for agent in agents],
            y=[pos[agent][1] for agent in agents],
            mode='markers+text',
            text=agents,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                size=30,
                color=[self.status_colors.get(agent_statuses[agent].get("status", "offline"), "#9E9E9E") 
                       for agent in agents],
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=edge_trace + [node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_activity_timeline(self, messages: List[Dict[str, Any]]):
        """
        Render activity timeline
        
        Args:
            messages: List of chat messages with timestamps
        """
        st.markdown("### 📅 Activity Timeline")
        
        if not messages:
            st.info("No activity yet. Start a conversation to see the timeline.")
            return
        
        # Create timeline data
        timeline_data = []
        for msg in messages[-10:]:  # Last 10 messages
            if "timestamp" in msg:
                timeline_data.append({
                    "Time": datetime.fromisoformat(msg["timestamp"]),
                    "Role": msg["role"].title(),
                    "Preview": msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
                })
        
        if timeline_data:
            df = pd.DataFrame(timeline_data)
            
            # Display as a simple timeline
            for _, row in df.iterrows():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.caption(row["Time"].strftime("%H:%M:%S"))
                with col2:
                    if row["Role"] == "User":
                        st.info(f"👤 {row['Preview']}")
                    else:
                        st.success(f"🤖 {row['Preview']}")
    
    def render_system_health(self, agent_statuses: Dict[str, Dict[str, Any]]):
        """
        Render system health indicators
        
        Args:
            agent_statuses: Dictionary of agent statuses
        """
        st.markdown("### 🏥 System Health")
        
        # Calculate health metrics
        total_agents = len(agent_statuses)
        active_agents = sum(1 for s in agent_statuses.values() if s.get("status") == "active")
        error_agents = sum(1 for s in agent_statuses.values() if s.get("status") == "error")
        
        health_score = (active_agents / total_agents * 100) if total_agents > 0 else 0
        
        # Health status
        if health_score >= 90:
            health_status = "Excellent"
            health_color = "green"
        elif health_score >= 70:
            health_status = "Good"
            health_color = "yellow"
        else:
            health_status = "Needs Attention"
            health_color = "red"
        
        # Display health score
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.metric(
                "System Health Score",
                f"{health_score:.0f}%",
                delta=f"{health_status}"
            )
            st.progress(health_score / 100)
        
        with col2:
            st.metric("Active", active_agents)
        
        with col3:
            st.metric("Errors", error_agents)
        
        # Health checks
        st.markdown("#### Health Checks")
        
        checks = {
            "Agent Connectivity": active_agents == total_agents,
            "Response Times": all(
                s.get("metrics", {}).get("avg_response_time", 0) < 5
                for s in agent_statuses.values()
            ),
            "Success Rates": all(
                s.get("metrics", {}).get("success_rate", 0) > 0.8
                for s in agent_statuses.values()
            ),
            "MCP Servers": True,  # Simplified for demo
            "Knowledge Base": True  # Simplified for demo
        }
        
        for check, status in checks.items():
            if status:
                st.success(f"✅ {check}")
            else:
                st.error(f"❌ {check}")


class AgentMetricsTracker:
    """
    Tracks and analyzes agent metrics over time
    """
    
    def __init__(self):
        # Initialize metrics storage
        if "metrics_history" not in st.session_state:
            st.session_state.metrics_history = []
    
    def record_metrics(self, agent_statuses: Dict[str, Dict[str, Any]]):
        """Record current metrics snapshot"""
        snapshot = {
            "timestamp": datetime.now(),
            "metrics": {}
        }
        
        for agent_name, status in agent_statuses.items():
            metrics = status.get("metrics", {})
            snapshot["metrics"][agent_name] = {
                "queries": metrics.get("queries_processed", 0),
                "success_rate": metrics.get("success_rate", 1.0),
                "response_time": metrics.get("avg_response_time", 0)
            }
        
        st.session_state.metrics_history.append(snapshot)
        
        # Keep only last 100 snapshots
        if len(st.session_state.metrics_history) > 100:
            st.session_state.metrics_history = st.session_state.metrics_history[-100:]
    
    def get_trend_data(self, metric_name: str, agent_name: str = None) -> pd.DataFrame:
        """Get trend data for a specific metric"""
        data = []
        
        for snapshot in st.session_state.metrics_history:
            timestamp = snapshot["timestamp"]
            
            if agent_name:
                # Specific agent
                if agent_name in snapshot["metrics"]:
                    value = snapshot["metrics"][agent_name].get(metric_name, 0)
                    data.append({
                        "Timestamp": timestamp,
                        "Value": value,
                        "Agent": agent_name
                    })
            else:
                # All agents
                for agent, metrics in snapshot["metrics"].items():
                    value = metrics.get(metric_name, 0)
                    data.append({
                        "Timestamp": timestamp,
                        "Value": value,
                        "Agent": agent
                    })
        
        return pd.DataFrame(data)
    
    def render_trends(self):
        """Render metric trends"""
        st.markdown("### 📊 Performance Trends")
        
        if len(st.session_state.metrics_history) < 2:
            st.info("Insufficient data for trends. Metrics will appear as the system processes queries.")
            return
        
        # Metric selection
        metric = st.selectbox(
            "Select Metric",
            ["queries", "success_rate", "response_time"],
            format_func=lambda x: {
                "queries": "Query Volume",
                "success_rate": "Success Rate",
                "response_time": "Response Time"
            }.get(x, x)
        )
        
        # Get trend data
        df = self.get_trend_data(metric)
        
        if not df.empty:
            # Create line chart
            fig = px.line(
                df,
                x="Timestamp",
                y="Value",
                color="Agent",
                title=f"{metric.replace('_', ' ').title()} Over Time",
                markers=True
            )
            
            # Customize layout
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title=metric.replace('_', ' ').title(),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)


def render_agent_comparison_table(agent_statuses: Dict[str, Dict[str, Any]]):
    """
    Render a comparison table of all agents
    
    Args:
        agent_statuses: Dictionary of agent statuses
    """
    st.markdown("### 📋 Agent Comparison")
    
    # Prepare data for table
    data = []
    for agent_name, status in agent_statuses.items():
        metrics = status.get("metrics", {})
        data.append({
            "Agent": agent_name,
            "Role": status.get("role", "Unknown"),
            "Status": status.get("status", "offline"),
            "Queries": metrics.get("queries_processed", 0),
            "Success %": f"{metrics.get('success_rate', 1.0) * 100:.1f}",
            "Avg Time (s)": f"{metrics.get('avg_response_time', 0):.2f}",
            "Capabilities": len(status.get("capabilities", []))
        })
    
    df = pd.DataFrame(data)
    
    # Style the dataframe
    def style_status(val):
        colors = {
            "active": "background-color: #e8f5e9",
            "busy": "background-color: #fff3e0",
            "error": "background-color: #ffebee",
            "offline": "background-color: #f5f5f5"
        }
        return colors.get(val, "")
    
    styled_df = df.style.applymap(style_status, subset=["Status"])
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)