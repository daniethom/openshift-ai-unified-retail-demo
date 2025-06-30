"""
Dashboard Page for Meridian Retail AI
Displays system overview and key metrics
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import random

# Import components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.agent_status import AgentStatusDisplay, AgentMetricsTracker

# Page config
st.set_page_config(
    page_title="Dashboard - Meridian Retail AI",
    page_icon="📊",
    layout="wide"
)

# Initialize components
status_display = AgentStatusDisplay()
metrics_tracker = AgentMetricsTracker()


def render_header():
    """Render page header"""
    st.title("📊 Meridian Retail AI Dashboard")
    st.markdown("Real-time monitoring of the multi-agent AI system")
    
    # Last update time
    col1, col2, col3 = st.columns([3, 1, 1])
    with col3:
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


def render_system_overview():
    """Render system overview section"""
    st.header("🎯 System Overview")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "System Status",
            "Operational",
            delta="All systems running"
        )
    
    with col2:
        queries_today = st.session_state.get("total_queries", 0)
        st.metric(
            "Queries Today",
            queries_today,
            delta=f"+{random.randint(5, 15)}"
        )
    
    with col3:
        avg_response = random.uniform(1.5, 3.0)
        st.metric(
            "Avg Response Time",
            f"{avg_response:.2f}s",
            delta="-0.3s"
        )
    
    with col4:
        success_rate = random.uniform(94, 99)
        st.metric(
            "Success Rate",
            f"{success_rate:.1f}%",
            delta="+1.2%"
        )
    
    with col5:
        uptime = 99.9
        st.metric(
            "Uptime",
            f"{uptime}%",
            delta="30 days"
        )


def render_query_analytics():
    """Render query analytics section"""
    st.header("📈 Query Analytics")
    
    # Create sample data for visualization
    hours = pd.date_range(
        start=datetime.now() - timedelta(hours=24),
        end=datetime.now(),
        freq='H'
    )
    
    query_data = pd.DataFrame({
        'Hour': hours,
        'Queries': [random.randint(10, 50) for _ in range(len(hours))],
        'Success': [random.randint(8, 48) for _ in range(len(hours))]
    })
    
    # Query volume chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=query_data['Hour'],
        y=query_data['Queries'],
        mode='lines+markers',
        name='Total Queries',
        line=dict(color='#1f77b4', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=query_data['Hour'],
        y=query_data['Success'],
        mode='lines+markers',
        name='Successful',
        line=dict(color='#2ca02c', width=2)
    ))
    
    fig.update_layout(
        title="Query Volume (Last 24 Hours)",
        xaxis_title="Time",
        yaxis_title="Number of Queries",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Query type breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        # Query types pie chart
        query_types = pd.DataFrame({
            'Type': ['Inventory', 'Pricing', 'Customer', 'Trends', 'Complex'],
            'Count': [120, 95, 85, 75, 50]
        })
        
        fig_pie = px.pie(
            query_types,
            values='Count',
            names='Type',
            title="Query Distribution by Type"
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Response time distribution
        response_times = pd.DataFrame({
            'Agent': ['HomeAgent', 'InventoryAgent', 'PricingAgent', 'CustomerAgent', 'TrendAgent'],
            'Avg Response Time': [2.1, 1.8, 2.3, 1.5, 2.7]
        })
        
        fig_bar = px.bar(
            response_times,
            x='Agent',
            y='Avg Response Time',
            title="Average Response Time by Agent",
            color='Avg Response Time',
            color_continuous_scale='Blues'
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)


def render_business_insights():
    """Render business insights section"""
    st.header("💡 Business Insights")
    
    # Insight cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🎯 Top Trending Query**
        
        "Winter fashion trends for Cape Town stores" 
        
        *Frequency: 15 times today*
        """)
    
    with col2:
        st.success("""
        **📈 Performance Highlight**
        
        Customer satisfaction predictions improved by 12% this week
        
        *Agent: CustomerAgent*
        """)
    
    with col3:
        st.warning("""
        **⚠️ Action Required**
        
        3 products below reorder threshold in Johannesburg
        
        *Agent: InventoryAgent*
        """)
    
    # Recent insights table
    st.subheader("Recent AI-Generated Insights")
    
    insights_data = pd.DataFrame({
        'Time': [
            datetime.now() - timedelta(minutes=5),
            datetime.now() - timedelta(minutes=15),
            datetime.now() - timedelta(minutes=30),
            datetime.now() - timedelta(minutes=45),
            datetime.now() - timedelta(hours=1)
        ],
        'Category': ['Inventory', 'Pricing', 'Trends', 'Customer', 'Complex'],
        'Insight': [
            'Stock levels for winter coats are 25% below optimal in Cape Town',
            'Competitor pricing analysis suggests 5-10% adjustment opportunity',
            'Gorpcore trend showing 40% growth in youth demographic',
            'VIP customer retention rate improved to 92% with personalized offers',
            'Cross-functional analysis identified R2.3M revenue opportunity'
        ],
        'Impact': ['High', 'Medium', 'Medium', 'High', 'Very High']
    })
    
    # Style the impact column
    def style_impact(val):
        colors = {
            'Very High': 'background-color: #d32f2f; color: white',
            'High': 'background-color: #f57c00; color: white',
            'Medium': 'background-color: #fbc02d',
            'Low': 'background-color: #689f38; color: white'
        }
        return colors.get(val, '')
    
    styled_df = insights_data.style.applymap(style_impact, subset=['Impact'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def render_agent_performance():
    """Render agent performance section"""
    st.header("🤖 Agent Performance")
    
    # Get agent statuses (mock data for demo)
    agent_statuses = {
        "HomeAgent": {
            "status": "active",
            "role": "Chief AI Orchestrator",
            "metrics": {
                "queries_processed": 425,
                "success_rate": 0.98,
                "avg_response_time": 2.1,
                "total_collaborations": 380
            },
            "capabilities": ["Query Analysis", "Agent Orchestration", "Response Synthesis"]
        },
        "InventoryAgent": {
            "status": "active",
            "role": "Inventory Management Specialist",
            "metrics": {
                "queries_processed": 120,
                "success_rate": 0.96,
                "avg_response_time": 1.8,
                "total_collaborations": 95
            },
            "capabilities": ["Stock Checking", "Demand Forecasting", "Reorder Planning"]
        },
        "PricingAgent": {
            "status": "busy",
            "role": "Revenue Optimization Specialist",
            "current_task": "Analyzing competitor pricing...",
            "metrics": {
                "queries_processed": 95,
                "success_rate": 0.94,
                "avg_response_time": 2.3,
                "total_collaborations": 85
            },
            "capabilities": ["Price Optimization", "Competitive Analysis", "Promotion Planning"]
        },
        "CustomerAgent": {
            "status": "active",
            "role": "Customer Experience Specialist",
            "metrics": {
                "queries_processed": 85,
                "success_rate": 0.97,
                "avg_response_time": 1.5,
                "total_collaborations": 70
            },
            "capabilities": ["Customer Support", "Personalization", "Loyalty Management"]
        },
        "TrendAgent": {
            "status": "active",
            "role": "Fashion Trend Analyst",
            "metrics": {
                "queries_processed": 75,
                "success_rate": 0.95,
                "avg_response_time": 2.7,
                "total_collaborations": 65
            },
            "capabilities": ["Trend Analysis", "Market Intelligence", "Seasonal Forecasting"]
        }
    }
    
    # Update session state
    if "agent_statuses" not in st.session_state:
        st.session_state.agent_statuses = agent_statuses
    
    # Display components
    tabs = st.tabs(["Status Overview", "Performance Charts", "Collaboration Network", "System Health"])
    
    with tabs[0]:
        status_display.render_agent_grid(agent_statuses)
    
    with tabs[1]:
        status_display.render_performance_chart(agent_statuses)
        metrics_tracker.record_metrics(agent_statuses)
        metrics_tracker.render_trends()
    
    with tabs[2]:
        status_display.render_collaboration_network(agent_statuses)
    
    with tabs[3]:
        status_display.render_system_health(agent_statuses)


def render_alerts_section():
    """Render alerts and notifications"""
    st.header("🔔 Alerts & Notifications")
    
    # Active alerts
    alerts = [
        {
            "level": "warning",
            "time": datetime.now() - timedelta(minutes=10),
            "message": "Response time for TrendAgent exceeding threshold",
            "action": "Monitoring performance"
        },
        {
            "level": "info",
            "time": datetime.now() - timedelta(minutes=30),
            "message": "New fashion trend detected: Sustainable Materials",
            "action": "Review trend report"
        },
        {
            "level": "success",
            "time": datetime.now() - timedelta(hours=1),
            "message": "System optimization completed successfully",
            "action": "Performance improved by 15%"
        }
    ]
    
    for alert in alerts:
        if alert["level"] == "warning":
            st.warning(f"**{alert['time'].strftime('%H:%M')}** - {alert['message']} | *{alert['action']}*")
        elif alert["level"] == "info":
            st.info(f"**{alert['time'].strftime('%H:%M')}** - {alert['message']} | *{alert['action']}*")
        else:
            st.success(f"**{alert['time'].strftime('%H:%M')}** - {alert['message']} | *{alert['action']}*")


def main():
    """Main dashboard logic"""
    render_header()
    
    # System overview
    render_system_overview()
    
    st.divider()
    
    # Query analytics
    render_query_analytics()
    
    st.divider()
    
    # Business insights
    render_business_insights()
    
    st.divider()
    
    # Agent performance
    render_agent_performance()
    
    st.divider()
    
    # Alerts
    render_alerts_section()
    
    # Auto-refresh
    if st.checkbox("Auto-refresh (10s)", value=False):
        st.empty()
        import time
        time.sleep(10)
        st.rerun()


if __name__ == "__main__":
    main()