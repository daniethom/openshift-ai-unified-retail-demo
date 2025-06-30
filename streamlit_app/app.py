"""
Meridian Retail AI Demo - Streamlit Application
Main entry point for the web interface
"""
import sys
import os

# Add the project root to the Python path
# This is necessary to ensure that the agent modules can be found
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import streamlit as st
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Meridian Retail AI Demo",
    page_icon="🏬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main content area */
    .main {
        padding: 1rem;
    }
    
    /* Chat messages */
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Agent status cards */
    .agent-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    /* Metrics */
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    /* Status indicators */
    .status-active {
        color: #4caf50;
        font-weight: bold;
    }
    
    .status-busy {
        color: #ff9800;
        font-weight: bold;
    }
    
    .status-error {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_system" not in st.session_state:
    st.session_state.agent_system = None

if "system_initialized" not in st.session_state:
    st.session_state.system_initialized = False

if "current_query_id" not in st.session_state:
    st.session_state.current_query_id = None


class AgentSystemInterface:
    """Interface to interact with the multi-agent system"""
    
    def __init__(self):
        self.agents = {}
        self.home_agent = None
        
    async def initialize(self):
        """Initialize the agent system"""
        try:
            # Import agent modules
            from agents.home_agent import HomeAgent
            from agents.inventory_agent import InventoryAgent
            from agents.pricing_agent import PricingAgent
            from agents.customer_agent import CustomerAgent
            from agents.trend_agent import TrendAgent
            
            # Mock MCP servers for demo
            mock_mcp_servers = {
                "llm_server": {"endpoint": "http://localhost:8001"},
                "rag_server": {"endpoint": "http://localhost:8002"},
                "search_server": {"endpoint": "http://localhost:8003"},
                "analytics_server": {"endpoint": "http://localhost:8004"}
            }
            
            # Mock data store
            mock_data_store = {}
            
            # Initialize agents
            self.agents = {
                "InventoryAgent": InventoryAgent(mock_mcp_servers, mock_data_store),
                "PricingAgent": PricingAgent(mock_mcp_servers, mock_data_store),
                "CustomerAgent": CustomerAgent(mock_mcp_servers, mock_data_store),
                "TrendAgent": TrendAgent(mock_mcp_servers, mock_data_store)
            }
            
            # Initialize home agent with agent registry
            self.home_agent = HomeAgent(mock_mcp_servers, self.agents)
            
            logger.info("Agent system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent system: {e}")
            return False
    
    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a query through the agent system"""
        if not self.home_agent:
            return {
                "status": "error",
                "error": "Agent system not initialized",
                "response": "I apologize, but the AI system is not ready yet. Please try again in a moment."
            }
        
        try:
            # Process through home agent
            result = await self.home_agent.process_query(query, context or {})
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "status": "error",
                "error": str(e),
                "response": "I encountered an error while processing your request. Please try again."
            }
    
    def get_agent_statuses(self) -> Dict[str, Any]:
        """Get status of all agents"""
        statuses = {}
        
        if self.home_agent:
            statuses["HomeAgent"] = self.home_agent.get_status_dict()
        
        for name, agent in self.agents.items():
            statuses[name] = agent.get_status_dict()
        
        return statuses


async def initialize_system():
    """Initialize the agent system"""
    with st.spinner("🚀 Initializing Meridian Retail AI System..."):
        agent_system = AgentSystemInterface()
        success = await agent_system.initialize()
        
        if success:
            st.session_state.agent_system = agent_system
            st.session_state.system_initialized = True
            st.success("✅ System initialized successfully!")
        else:
            st.error("❌ Failed to initialize system. Please refresh the page.")
        
        return success


def display_header():
    """Display application header"""
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.title("🏬 Meridian Retail AI Demo")
        st.markdown("**Multi-Agent AI System for Retail Operations**")
    
    with col3:
        if st.session_state.system_initialized:
            st.markdown("### System Status")
            st.markdown('<span class="status-active">● Active</span>', unsafe_allow_html=True)


def display_sidebar():
    """Display sidebar with agent status and controls"""
    with st.sidebar:
        st.header("🤖 AI Agents")
        
        if st.session_state.system_initialized and st.session_state.agent_system:
            # Get agent statuses
            statuses = st.session_state.agent_system.get_agent_statuses()
            
            # Display each agent's status
            for agent_name, status in statuses.items():
                with st.expander(f"{agent_name}", expanded=False):
                    status_color = {
                        "active": "status-active",
                        "busy": "status-busy",
                        "error": "status-error"
                    }.get(status.get("status", "active"), "status-active")
                    
                    st.markdown(f'<span class="{status_color}">● {status.get("status", "unknown")}</span>', 
                              unsafe_allow_html=True)
                    
                    # Display metrics
                    metrics = status.get("metrics", {})
                    if metrics:
                        st.metric("Queries", metrics.get("queries_processed", 0))
                        st.metric("Success Rate", f"{metrics.get('success_rate', 0) * 100:.1f}%")
        else:
            st.info("System initializing...")
        
        st.divider()
        
        # Demo scenarios
        st.header("📋 Demo Scenarios")
        
        scenarios = {
            "🎯 Fashion Trend Analysis": "What winter fashion trends should our Cape Town stores focus on for professional women?",
            "🛍️ Cross-Sell Opportunity": "Customer Sarah Johnson bought a winter coat. What should we recommend?",
            "📊 Inventory Optimization": "Optimize inventory for the upcoming summer season across all Johannesburg stores.",
            "🤝 Customer Service": "A high-value customer is complaining about a delayed delivery and poor service."
        }
        
        for title, query in scenarios.items():
            if st.button(title, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
        
        st.divider()
        
        # System info
        st.header("ℹ️ System Info")
        st.info("""
        This demo showcases:
        - Multi-agent orchestration
        - MCP protocol integration
        - RAG capabilities
        - Real-time collaboration
        """)


def display_chat_interface():
    """Display the main chat interface"""
    # Chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 **You:** {message["content"]}</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">🤖 **AI Assistant:** {message["content"]}</div>', 
                          unsafe_allow_html=True)
                
                # Display additional details if available
                if "details" in message:
                    with st.expander("View Details"):
                        st.json(message["details"])
    
    # Input area
    st.divider()
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask a question about retail operations...",
            placeholder="e.g., What are the current inventory levels for winter coats?",
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    # Process input
    if send_button and user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Process query
        asyncio.run(process_user_query(user_input))
        
        # Clear input and rerun
        st.rerun()


async def process_user_query(query: str):
    """Process user query through the agent system"""
    if not st.session_state.system_initialized or not st.session_state.agent_system:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "The system is still initializing. Please wait a moment and try again."
        })
        return
    
    # Show processing indicator
    with st.spinner("🤔 AI agents are collaborating on your query..."):
        # Process through agent system
        result = await st.session_state.agent_system.process_query(query)
        
        # Extract response
        if result.get("status") == "success":
            response = result.get("response", {})
            
            # Format the response
            if isinstance(response, dict):
                summary = response.get("summary", "I've analyzed your query.")
                insights = response.get("detailed_insights", {})
                recommendations = response.get("recommendations", [])
                
                # Build formatted response
                formatted_response = summary
                
                if insights:
                    formatted_response += "\n\n**Key Insights:**\n"
                    for agent, insight in insights.items():
                        if isinstance(insight, dict) and insight.get("result"):
                            formatted_response += f"- {agent}: {insight['result']}\n"
                
                if recommendations:
                    formatted_response += "\n**Recommendations:**\n"
                    for i, rec in enumerate(recommendations, 1):
                        formatted_response += f"{i}. {rec}\n"
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": formatted_response,
                    "details": result
                })
            else:
                # Simple response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": str(response),
                    "details": result
                })
        else:
            # Error response
            error_msg = result.get("error", "Unknown error occurred")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I encountered an error: {error_msg}. Please try rephrasing your question.",
                "details": result
            })


def display_metrics_dashboard():
    """Display key metrics dashboard"""
    st.header("📊 Real-Time Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Active Agents",
            value="5",
            delta="All operational"
        )
    
    with col2:
        st.metric(
            label="Queries Processed",
            value=len(st.session_state.messages) // 2,
            delta="+1" if st.session_state.messages else None
        )
    
    with col3:
        st.metric(
            label="Avg Response Time",
            value="2.3s",
            delta="-0.5s"
        )
    
    with col4:
        st.metric(
            label="System Health",
            value="98%",
            delta="+2%"
        )


def main():
    """Main application logic"""
    # Display header
    display_header()
    
    # Initialize system if needed
    if not st.session_state.system_initialized:
        asyncio.run(initialize_system())
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Dashboard", "🔧 Settings"])
    
    with tab1:
        display_chat_interface()
    
    with tab2:
        display_metrics_dashboard()
        
        # Additional analytics
        st.header("🎯 Agent Performance")
        
        if st.session_state.system_initialized and st.session_state.agent_system:
            agent_statuses = st.session_state.agent_system.get_agent_statuses()
            
            # Create columns for agent cards
            cols = st.columns(2)
            
            for i, (agent_name, status) in enumerate(agent_statuses.items()):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="agent-card">
                        <h4>{agent_name}</h4>
                        <p><strong>Role:</strong> {status.get('role', 'Unknown')}</p>
                        <p><strong>Status:</strong> {status.get('status', 'Unknown')}</p>
                        <p><strong>Capabilities:</strong> {len(status.get('capabilities', []))}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab3:
        st.header("⚙️ System Settings")
        
        # Model settings
        st.subheader("🧠 Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.selectbox(
                "LLM Model",
                ["granite-3b", "llama-3.2", "gpt-3.5-turbo"],
                index=0
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1
            )
        
        with col2:
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=100,
                max_value=4000,
                value=2048,
                step=100
            )
            
            top_k = st.number_input(
                "RAG Top-K Results",
                min_value=1,
                max_value=10,
                value=5
            )
        
        # Agent settings
        st.subheader("🤖 Agent Configuration")
        
        enable_parallel = st.checkbox("Enable Parallel Agent Execution", value=True)
        enable_caching = st.checkbox("Enable Response Caching", value=True)
        
        # Save settings button
        if st.button("💾 Save Settings", type="primary"):
            st.success("Settings saved successfully!")
            
        # Clear chat button
        st.divider()
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()