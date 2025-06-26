"""
Base Agent Framework for Unified Retail AI System
Provides common functionality for all specialized agents
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field
from crewai import Agent as CrewAIAgent

# Configure logging
logger = logging.getLogger(__name__)


class AgentMessage(BaseModel):
    """Standard message format for inter-agent communication"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    sender: str
    recipient: str
    message_type: str  # query, response, notification, error
    content: Any
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentCapability(BaseModel):
    """Defines a capability that an agent can provide"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    examples: List[Dict[str, Any]] = Field(default_factory=list)


class AgentStatus(BaseModel):
    """Agent status information"""
    agent_id: str
    name: str
    status: str  # active, busy, error, offline
    current_task: Optional[str] = None
    metrics: Dict[str, Any]
    last_updated: datetime = Field(default_factory=datetime.now)


class BaseAgent(ABC):
    """
    Abstract base class for all retail AI agents
    Provides common functionality and interface
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        backstory: str,
        mcp_servers: Optional[Dict[str, Any]] = None,
        capabilities: Optional[List[AgentCapability]] = None,
        verbose: bool = True
    ):
        self.id = str(uuid4())
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.mcp_servers = mcp_servers or {}
        self.capabilities = capabilities or []
        self.verbose = verbose
        
        # Agent state
        self.status = "active"
        self.current_task = None
        
        # Message history
        self.message_history: List[AgentMessage] = []
        self.max_history_size = 1000
        
        # Performance metrics
        self.metrics = {
            "queries_processed": 0,
            "avg_response_time": 0,
            "success_rate": 1.0,
            "last_active": datetime.now(),
            "total_collaborations": 0,
            "errors": 0
        }
        
        # Collaboration partners
        self.known_agents: Dict[str, 'BaseAgent'] = {}
        
        # Initialize CrewAI agent
        self._crew_agent = self._create_crew_agent()
        
        logger.info(f"Initialized {self.name} agent with role: {self.role}")
    
    def _create_crew_agent(self) -> CrewAIAgent:
        """Create the underlying CrewAI agent"""
        return CrewAIAgent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            verbose=self.verbose,
            allow_delegation=True,
            max_iter=5
        )
    
    @abstractmethod
    async def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query and return results
        Must be implemented by each specialized agent
        """
        pass
    
    @abstractmethod
    def get_tools(self) -> List[Any]:
        """
        Return the list of tools available to this agent
        Must be implemented by each specialized agent
        """
        pass
    
    def register_agent(self, agent: 'BaseAgent') -> None:
        """Register another agent for collaboration"""
        self.known_agents[agent.name] = agent
        logger.debug(f"{self.name} registered collaboration partner: {agent.name}")
    
    async def send_message(
        self,
        recipient: str,
        content: Any,
        message_type: str = "query",
        correlation_id: Optional[str] = None
    ) -> AgentMessage:
        """Send a message to another agent"""
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id
        )
        
        self._add_to_history(message)
        logger.debug(f"{self.name} sending {message_type} to {recipient}")
        
        # If we know the recipient agent, deliver directly
        if recipient in self.known_agents:
            response = await self.known_agents[recipient].receive_message(message)
            if response:
                self._add_to_history(response)
            return response or message
        
        return message
    
    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process a received message and optionally return a response"""
        self._add_to_history(message)
        logger.debug(f"{self.name} received {message.message_type} from {message.sender}")
        
        if message.message_type == "query":
            # Process the query and send response
            try:
                self.status = "busy"
                self.current_task = f"Processing query from {message.sender}"
                
                result = await self.process_query(
                    message.content.get("query", ""),
                    message.content.get("context", {})
                )
                
                response = AgentMessage(
                    sender=self.name,
                    recipient=message.sender,
                    message_type="response",
                    content=result,
                    correlation_id=message.id
                )
                
                self.status = "active"
                self.current_task = None
                self.metrics["total_collaborations"] += 1
                
                return response
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                self.metrics["errors"] += 1
                self.status = "error"
                
                error_response = AgentMessage(
                    sender=self.name,
                    recipient=message.sender,
                    message_type="error",
                    content={"error": str(e), "error_type": type(e).__name__},
                    correlation_id=message.id
                )
                
                self.status = "active"
                self.current_task = None
                
                return error_response
        
        elif message.message_type == "notification":
            # Handle notifications
            logger.info(f"{self.name} received notification: {message.content}")
            
        return None
    
    def _add_to_history(self, message: AgentMessage) -> None:
        """Add message to history with size limit"""
        self.message_history.append(message)
        
        # Maintain history size limit
        if len(self.message_history) > self.max_history_size:
            self.message_history = self.message_history[-self.max_history_size:]
    
    def update_metrics(self, response_time: float, success: bool = True) -> None:
        """Update agent performance metrics"""
        self.metrics["queries_processed"] += 1
        self.metrics["last_active"] = datetime.now()
        
        # Update average response time (exponential moving average)
        alpha = 0.1  # Smoothing factor
        self.metrics["avg_response_time"] = (
            alpha * response_time + 
            (1 - alpha) * self.metrics["avg_response_time"]
        )
        
        # Update success rate
        if not success:
            self.metrics["errors"] += 1
        
        total_attempts = self.metrics["queries_processed"]
        successful_attempts = total_attempts - self.metrics["errors"]
        self.metrics["success_rate"] = successful_attempts / total_attempts if total_attempts > 0 else 1.0
    
    def get_status(self) -> AgentStatus:
        """Get current agent status"""
        return AgentStatus(
            agent_id=self.id,
            name=self.name,
            status=self.status,
            current_task=self.current_task,
            metrics=self.metrics
        )
    
    def get_status_dict(self) -> Dict[str, Any]:
        """Get current agent status as dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "status": self.status,
            "current_task": self.current_task,
            "metrics": self.metrics,
            "capabilities": [cap.name for cap in self.capabilities],
            "mcp_servers": list(self.mcp_servers.keys()),
            "known_agents": list(self.known_agents.keys())
        }
    
    def describe_capabilities(self) -> List[Dict[str, Any]]:
        """Return a description of agent capabilities"""
        return [
            {
                "name": cap.name,
                "description": cap.description,
                "input": cap.input_schema,
                "output": cap.output_schema,
                "examples": cap.examples
            }
            for cap in self.capabilities
        ]
    
    async def collaborate_with(
        self,
        other_agent: Union[str, 'BaseAgent'],
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collaborate with another agent on a query"""
        # Determine target agent
        if isinstance(other_agent, str):
            if other_agent not in self.known_agents:
                return {
                    "error": f"Unknown agent: {other_agent}",
                    "known_agents": list(self.known_agents.keys())
                }
            target_agent = self.known_agents[other_agent]
        else:
            target_agent = other_agent
        
        # Send collaboration request
        message = await self.send_message(
            recipient=target_agent.name,
            content={
                "query": query,
                "context": {
                    **context,
                    "requesting_agent": self.name,
                    "collaboration_request": True
                }
            },
            message_type="query"
        )
        
        # Wait for response (in production, this would be async with timeout)
        # For now, we'll get the response directly
        if hasattr(message, 'content') and message.message_type == "response":
            return message.content
        
        return {"error": "No response from collaborating agent"}
    
    def get_conversation_history(
        self, 
        partner: Optional[str] = None,
        limit: int = 10
    ) -> List[AgentMessage]:
        """Get conversation history, optionally filtered by partner"""
        if partner:
            filtered_history = [
                msg for msg in self.message_history
                if msg.sender == partner or msg.recipient == partner
            ]
            return filtered_history[-limit:]
        
        return self.message_history[-limit:]
    
    def clear_history(self) -> None:
        """Clear message history"""
        self.message_history.clear()
        logger.info(f"{self.name} cleared message history")
    
    def reset_metrics(self) -> None:
        """Reset performance metrics"""
        self.metrics = {
            "queries_processed": 0,
            "avg_response_time": 0,
            "success_rate": 1.0,
            "last_active": datetime.now(),
            "total_collaborations": 0,
            "errors": 0
        }
        logger.info(f"{self.name} reset metrics")
    
    async def broadcast_message(
        self,
        content: Any,
        message_type: str = "notification"
    ) -> List[AgentMessage]:
        """Broadcast a message to all known agents"""
        responses = []
        
        for agent_name in self.known_agents:
            response = await self.send_message(
                recipient=agent_name,
                content=content,
                message_type=message_type
            )
            responses.append(response)
        
        return responses
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of agent performance"""
        return {
            "agent": self.name,
            "role": self.role,
            "total_queries": self.metrics["queries_processed"],
            "success_rate": f"{self.metrics['success_rate'] * 100:.1f}%",
            "avg_response_time": f"{self.metrics['avg_response_time']:.2f}s",
            "collaborations": self.metrics["total_collaborations"],
            "errors": self.metrics["errors"],
            "uptime": (datetime.now() - self.metrics["last_active"]).total_seconds(),
            "status": self.status
        }
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', role='{self.role}', status='{self.status}')>"
    
    def __str__(self) -> str:
        return f"{self.name} ({self.role})"
