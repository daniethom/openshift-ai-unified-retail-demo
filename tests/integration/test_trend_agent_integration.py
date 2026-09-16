import os
import sys
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
import uvicorn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.trend_agent import TrendAgent
from mcp_servers.search_server import app as search_app


class UvicornTestServer(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    def run_in_thread(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        while not self.started:
            time.sleep(1e-3)

    def stop(self):
        self.should_exit = True
        self.thread.join()


@pytest.fixture(scope="module")
def live_search_server():
    host = "127.0.0.1"
    port = 8003
    config = uvicorn.Config(search_app, host=host, port=port, log_level="warning")
    server = UvicornTestServer(config=config)
    server.run_in_thread()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def trend_agent_for_integration(live_search_server):
    agent = TrendAgent(mcp_servers={}, data_store=MagicMock())
    agent.trend_searcher.search_fashion_trends = AsyncMock(
        side_effect=lambda query, location="national": {
            "trends": [
                {
                    "name": "Utility & Gorpcore",
                    "relevance": 0.92,
                    "growth": "rising",
                    "source": live_search_server,
                }
            ]
        }
    )
    agent.rag_retriever.retrieve = AsyncMock(return_value=[])
    return agent


@pytest.mark.asyncio
async def test_trend_agent_integrates_with_search_server(trend_agent_for_integration):
    query = "Analyze summer trends for Cape Town"
    context = {"season": "summer", "region": "Cape Town", "category": "womenswear"}

    result = await trend_agent_for_integration.process_query(query, context)

    assert result["status"] == "success"
    assert "analysis" in result
    trend_agent_for_integration.trend_searcher.search_fashion_trends.assert_awaited_once()
