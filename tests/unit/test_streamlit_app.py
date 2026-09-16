import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def test_streamlit_app_imports():
    import streamlit_app.app as app_module

    assert hasattr(app_module, "AgentSystemInterface")


@pytest.mark.asyncio
async def test_agent_system_interface_initialize():
    from streamlit_app.app import AgentSystemInterface

    mock_home = MagicMock()
    with (
        patch("agents.home_agent.HomeAgent", return_value=mock_home),
        patch("agents.inventory_agent.InventoryAgent", return_value=MagicMock()),
        patch("agents.pricing_agent.PricingAgent", return_value=MagicMock()),
        patch("agents.customer_agent.CustomerAgent", return_value=MagicMock()),
        patch("agents.trend_agent.TrendAgent", return_value=MagicMock()),
    ):
        interface = AgentSystemInterface()
        success = await interface.initialize()

    assert success is True
    assert interface.home_agent is mock_home


@pytest.mark.asyncio
async def test_agent_system_interface_process_query_without_init():
    from streamlit_app.app import AgentSystemInterface

    interface = AgentSystemInterface()
    result = await interface.process_query("hello", {})

    assert result["status"] == "error"
    assert "not initialized" in result["error"].lower()


@pytest.mark.asyncio
async def test_agent_system_interface_process_query_delegates_to_home_agent():
    from streamlit_app.app import AgentSystemInterface

    interface = AgentSystemInterface()
    interface.home_agent = MagicMock()
    interface.home_agent.process_query = AsyncMock(return_value={"status": "success"})

    result = await interface.process_query("stock for MF001", {})

    assert result["status"] == "success"
    interface.home_agent.process_query.assert_awaited_once()
