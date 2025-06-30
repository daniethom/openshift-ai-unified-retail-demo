"""
Chat Interface Component for Meridian Retail AI
Handles the chat UI and message rendering
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from datetime import datetime
import json


class ChatInterface:
    """
    Manages the chat interface for the Streamlit app
    """
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize chat-related session state"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "chat_input_key" not in st.session_state:
            st.session_state.chat_input_key = 0
    
    def render_message(self, message: Dict[str, Any]):
        """
        Render a single chat message
        
        Args:
            message: Message dictionary with role, content, and optional details
        """
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                # Main response
                st.markdown(message["content"])
                
                # Additional details (if available)
                if "details" in message:
                    self._render_message_details(message["details"])
    
    def _render_message_details(self, details: Dict[str, Any]):
        """Render additional message details in an expandable section"""
        with st.expander("🔍 View Details"):
            # Show which agents were involved
            if "metadata" in details and "agents_used" in details["metadata"]:
                st.markdown("**Agents Involved:**")
                agents = details["metadata"]["agents_used"]
                cols = st.columns(len(agents))
                for i, agent in enumerate(agents):
                    with cols[i]:
                        st.info(f"✓ {agent}")
            
            # Show response time
            if "metadata" in details and "response_time" in details["metadata"]:
                st.metric("Response Time", f"{details['metadata']['response_time']:.2f}s")
            
            # Show confidence level
            if "response" in details and "confidence_level" in details["response"]:
                confidence = details["response"]["confidence_level"]
                st.progress(confidence)
                st.caption(f"Confidence: {confidence:.0%}")
            
            # Show raw data (collapsible)
            if st.checkbox("Show Raw Response", key=f"raw_{id(details)}"):
                st.json(details)
    
    def render_chat_history(self):
        """Render the complete chat history"""
        for message in st.session_state.messages:
            self.render_message(message)
    
    def render_input_area(self) -> Optional[str]:
        """
        Render the chat input area
        
        Returns:
            User input if submitted, None otherwise
        """
        # Create a form for better control
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([6, 1])
            
            with col1:
                user_input = st.text_input(
                    "Message",
                    placeholder="Ask about inventory, pricing, trends, or customers...",
                    label_visibility="collapsed",
                    key=f"chat_input_{st.session_state.chat_input_key}"
                )
            
            with col2:
                submit = st.form_submit_button("Send", type="primary", use_container_width=True)
            
            if submit and user_input:
                # Increment key to clear input
                st.session_state.chat_input_key += 1
                return user_input
        
        return None
    
    def add_message(self, role: str, content: str, details: Optional[Dict[str, Any]] = None):
        """
        Add a message to the chat history
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            details: Optional additional details
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if details:
            message["details"] = details
        
        st.session_state.messages.append(message)
    
    def clear_history(self):
        """Clear the chat history"""
        st.session_state.messages = []
        st.session_state.chat_input_key += 1
    
    def render_quick_actions(self):
        """Render quick action buttons"""
        st.markdown("### 🚀 Quick Actions")
        
        quick_queries = {
            "📊 Stock Check": "What are the current stock levels for winter coats?",
            "💰 Price Analysis": "Analyze pricing for our top-selling products",
            "📈 Trend Report": "What are the latest fashion trends for summer?",
            "👥 Customer Insights": "Show me our top customers and their preferences"
        }
        
        cols = st.columns(2)
        for i, (label, query) in enumerate(quick_queries.items()):
            with cols[i % 2]:
                if st.button(label, use_container_width=True):
                    return query
        
        return None
    
    def render_typing_indicator(self):
        """Show a typing indicator while processing"""
        with st.chat_message("assistant", avatar="🤖"):
            with st.empty():
                for i in range(3):
                    dots = "." * ((i % 3) + 1)
                    st.write(f"Thinking{dots}")
                    st.sleep(0.3)
                st.write("")
    
    def format_agent_response(self, response: Dict[str, Any]) -> str:
        """
        Format the agent response for display
        
        Args:
            response: Raw response from agent system
            
        Returns:
            Formatted response string
        """
        if response.get("status") == "error":
            return f"❌ Error: {response.get('error', 'Unknown error occurred')}"
        
        # Extract main response
        result = response.get("response", {})
        
        if isinstance(result, str):
            return result
        
        # Format structured response
        formatted = ""
        
        # Summary
        if "summary" in result:
            formatted += result["summary"] + "\n\n"
        
        # Key insights
        if "detailed_insights" in result:
            formatted += "**📊 Key Insights:**\n"
            for agent, insight in result["detailed_insights"].items():
                if isinstance(insight, dict):
                    insight_text = insight.get("result", insight.get("summary", str(insight)))
                else:
                    insight_text = str(insight)
                formatted += f"- **{agent}**: {insight_text}\n"
            formatted += "\n"
        
        # Recommendations
        if "recommendations" in result:
            formatted += "**💡 Recommendations:**\n"
            for i, rec in enumerate(result["recommendations"], 1):
                if isinstance(rec, dict):
                    rec_text = rec.get("action", rec.get("description", str(rec)))
                else:
                    rec_text = str(rec)
                formatted += f"{i}. {rec_text}\n"
            formatted += "\n"
        
        # Confidence
        if "confidence_level" in result:
            confidence = result["confidence_level"]
            formatted += f"\n*Confidence: {confidence:.0%}*"
        
        return formatted.strip() or "I've processed your request successfully."


class MessageFormatter:
    """
    Formats messages for different types of agent responses
    """
    
    @staticmethod
    def format_inventory_response(data: Dict[str, Any]) -> str:
        """Format inventory-related responses"""
        formatted = "**📦 Inventory Status:**\n\n"
        
        if "stock_levels" in data:
            for location, levels in data["stock_levels"].items():
                formatted += f"**{location.title()}:**\n"
                formatted += f"- On Hand: {levels.get('on_hand', 0)} units\n"
                formatted += f"- Available: {levels.get('available', 0)} units\n"
                formatted += f"- Incoming: {levels.get('incoming', 0)} units\n\n"
        
        if "reorder_status" in data:
            status = data["reorder_status"]
            if status.get("needs_reorder"):
                formatted += f"⚠️ **Reorder Alert:** {status.get('urgency', 'Medium')} priority\n"
                formatted += f"Days until stockout: {status.get('days_until_stockout', 'Unknown')}\n"
        
        return formatted
    
    @staticmethod
    def format_pricing_response(data: Dict[str, Any]) -> str:
        """Format pricing-related responses"""
        formatted = "**💰 Pricing Analysis:**\n\n"
        
        if "current_pricing" in data:
            pricing = data["current_pricing"]
            formatted += f"Current Price: R{pricing.get('current_price', 0):,.2f}\n"
            formatted += f"Cost: R{pricing.get('cost', 0):,.2f}\n"
            formatted += f"Margin: {pricing.get('margin', 0) * 100:.1f}%\n\n"
        
        if "optimization" in data:
            opt = data["optimization"]
            formatted += "**Recommended Changes:**\n"
            formatted += f"- New Price: R{opt.get('recommended_price', 0):,.2f}\n"
            formatted += f"- Expected Margin: {opt.get('new_margin', 0) * 100:.1f}%\n"
            formatted += f"- Confidence: {opt.get('confidence', 0) * 100:.0f}%\n"
        
        return formatted
    
    @staticmethod
    def format_trend_response(data: Dict[str, Any]) -> str:
        """Format trend-related responses"""
        formatted = "**👗 Trend Analysis:**\n\n"
        
        if "top_trends" in data:
            formatted += "**Top Trends:**\n"
            for i, trend in enumerate(data["top_trends"][:5], 1):
                formatted += f"{i}. **{trend.get('name', 'Unknown')}**\n"
                formatted += f"   - Relevance: {trend.get('relevance', 0) * 100:.0f}%\n"
                formatted += f"   - Growth: {trend.get('growth', 'Stable')}\n\n"
        
        if "recommendations" in data:
            formatted += "**Strategic Recommendations:**\n"
            for rec in data["recommendations"]:
                formatted += f"- {rec}\n"
        
        return formatted
    
    @staticmethod
    def format_customer_response(data: Dict[str, Any]) -> str:
        """Format customer-related responses"""
        formatted = "**👥 Customer Service Response:**\n\n"
        
        if "response" in data:
            formatted += f"{data['response']}\n\n"
        
        if "suggested_actions" in data:
            formatted += "**Suggested Actions:**\n"
            for action in data["suggested_actions"]:
                formatted += f"- {action}\n"
        
        if "satisfaction_predicted" in data:
            satisfaction = data["satisfaction_predicted"]
            formatted += f"\n*Predicted Satisfaction: {satisfaction * 100:.0f}%*"
        
        return formatted