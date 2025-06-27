import pytest
import asyncio
from typing import Dict, Any, List
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents.base_agent import BaseAgent, AgentMessage

# --- Mock Implementation for Testing ---

class MockConcreteAgent(BaseAgent):
    """
    A concrete implementation of BaseAgent for testing purposes.
    It implements the abstract methods required by the base class.
    """
    async def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in query:
            raise ValueError("This is a simulated processing error.")
        
        response_content = f"Processed query: '{query}'"
        return {"response": response_content}

    def get_tools(self) -> List[Any]:
        return []

# --- Pytest Fixtures ---

@pytest.fixture
def agent_pair():
    """
    A fixture that provides two mock agents that are aware of each other,
    setting the stage for communication tests.
    """
    agent_a = MockConcreteAgent(name="AgentA", role="TesterA", goal="Test A", backstory="...")
    agent_b = MockConcreteAgent(name="AgentB", role="TesterB", goal="Test B", backstory="...")
    
    agent_a.register_agent(agent_b)
    agent_b.register_agent(agent_a)
    
    return agent_a, agent_b

# --- Test Cases ---

def test_agent_initialization():
    """
    Tests that the agent is initialized with the correct default state and metrics.
    """
    # Arrange
    agent = MockConcreteAgent(name="TestAgent", role="Tester", goal="Test Goal", backstory="...")
    
    # Assert
    assert agent.name == "TestAgent"
    assert agent.status == "active"
    assert agent.metrics["queries_processed"] == 0
    assert agent.metrics["success_rate"] == 1.0
    assert len(agent.message_history) == 0

def test_register_agent(agent_pair):
    """
    Tests that one agent can be successfully registered with another.
    """
    # Arrange
    agent_a, agent_b = agent_pair
    
    # Assert
    assert "AgentB" in agent_a.known_agents
    assert "AgentA" in agent_b.known_agents
    assert agent_a.known_agents["AgentB"] == agent_b

def test_update_metrics():
    """
    Tests that the performance metrics are updated correctly after a simulated query.
    """
    # Arrange
    agent = MockConcreteAgent(name="MetricAgent", role="Tester", goal="Test Goal", backstory="...")
    
    # Act
    agent.update_metrics(response_time=0.5, success=True)
    agent.update_metrics(response_time=1.5, success=False)
    
    # Assert
    assert agent.metrics["queries_processed"] == 2
    assert agent.metrics["errors"] == 1
    assert agent.metrics["success_rate"] == 0.5
    assert agent.metrics["avg_response_time"] > 0

@pytest.mark.asyncio
async def test_send_and_receive_message_query(agent_pair):
    """
    An async test that simulates one agent sending a query to another and
    verifies that a response is generated and received correctly.
    """
    # Arrange
    agent_a, agent_b = agent_pair
    query_content = "Can you process this for me?"
    
    # Act: Agent A sends a message to Agent B
    response_message = await agent_a.send_message(
        recipient="AgentB",
        content={"query": query_content}
    )
    
    # Assert
    assert response_message.message_type == "response"
    assert response_message.sender == "AgentB"
    assert response_message.recipient == "AgentA"
    assert f"Processed query: '{query_content}'" in response_message.content["response"]
    
    # Check history of both agents
    assert len(agent_a.message_history) == 2 # Original message + response
    assert len(agent_b.message_history) == 1 # Received message

@pytest.mark.asyncio
async def test_receive_message_error_handling(agent_pair):
    """
    Tests that if process_query raises an exception, an 'error' message
    is correctly sent back to the original sender.
    """
    # Arrange
    agent_a, agent_b = agent_pair
    error_query = "this query will cause an error"
    
    # Act
    error_response = await agent_a.send_message(
        recipient="AgentB",
        content={"query": error_query}
    )
    
    # Assert
    assert error_response.message_type == "error"
    assert error_response.sender == "AgentB"
    assert "This is a simulated processing error." in error_response.content["error"]
    assert agent_b.metrics["errors"] == 1
    assert agent_b.status == "active" # Status should reset after error

def test_message_history_limit():
    """
    Tests that the message history does not grow beyond the defined max_history_size.
    """
    # Arrange
    agent = MockConcreteAgent(name="HistoryAgent", role="Tester", goal="Test", backstory="...")
    agent.max_history_size = 5 # Set a small limit for testing
    
    # Act
    for i in range(10):
        msg = AgentMessage(sender="Test", recipient=agent.name, message_type="notification", content=f"Msg {i}")
        agent._add_to_history(msg)
        
    # Assert
    assert len(agent.message_history) == 5
    assert agent.message_history[0].content == "Msg 5" # Check the first message is the correct one
