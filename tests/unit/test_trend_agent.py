import pytest
from unittest.mock import MagicMock, AsyncMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents.trend_agent import TrendAgent


@pytest.fixture
def mock_data_store():
    return MagicMock()


@pytest.fixture
def trend_agent(mock_data_store):
    return TrendAgent(mcp_servers={}, data_store=mock_data_store)


@pytest.mark.parametrize("query, expected_type", [
    ("what are the winter trends?", "seasonal_analysis"),
    ("analyze emerging micro trends", "micro_macro"),
    ("what is zara doing with utility wear?", "competitor_watch"),
    ("general trend analysis", "general_analysis")
])
def test_classify_trend_query(trend_agent, query, expected_type):
    query_type = trend_agent._classify_trend_query(query)
    assert query_type == expected_type


@pytest.mark.asyncio
async def test_process_query_handles_seasonal_analysis(trend_agent, monkeypatch):
    query = "What are the key trends for summer in Cape Town?"
    context = {"season": "summer", "region": "Cape Town", "category": "womenswear"}

    mock_search = AsyncMock(
        return_value={"trends": [{"name": "mock_trend", "relevance": 0.9, "growth": "rising"}]}
    )
    mock_rag = AsyncMock(return_value=[{"source": "mock_report.pdf", "content": "test"}])

    monkeypatch.setattr(trend_agent.trend_searcher, "search_fashion_trends", mock_search)
    monkeypatch.setattr(trend_agent.rag_retriever, "retrieve", mock_rag)

    result = await trend_agent.process_query(query, context)

    assert result["status"] == "success"
    assert "analysis" in result
    mock_search.assert_awaited_once()
    mock_rag.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_query_routes_to_competitor_watch(trend_agent, monkeypatch):
    query = "What is H&M doing with Gorpcore?"
    context = {}

    mock_handler = AsyncMock(return_value={"competitor_adoption": []})
    monkeypatch.setattr(trend_agent, '_handle_competitor_watch', mock_handler)

    await trend_agent.process_query(query, context)

    mock_handler.assert_awaited_once_with(query, context)


@pytest.mark.asyncio
async def test_process_query_general_error_handling(trend_agent, monkeypatch):
    query = "This query is destined to fail."

    monkeypatch.setattr(
        trend_agent,
        '_classify_trend_query',
        MagicMock(side_effect=Exception("Classification failed")),
    )

    result = await trend_agent.process_query(query, {})

    assert result["status"] == "error"
    assert "Classification failed" in result["error"]
