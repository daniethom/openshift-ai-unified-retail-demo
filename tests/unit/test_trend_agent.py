import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.trend_agent import TrendAgent


@pytest.fixture
def trend_agent():
    return TrendAgent(mcp_servers={}, data_store=MagicMock())


@pytest.mark.asyncio
async def test_process_query_success(trend_agent, monkeypatch):
    monkeypatch.setattr(
        trend_agent,
        "_gather_trend_data",
        AsyncMock(
            return_value={
                "current_trends": {
                    "trends": [{"name": "Utility Wear", "relevance": 0.9}]
                },
                "historical_patterns": [],
                "competitor_insights": {},
                "social_media": {},
            }
        ),
    )
    monkeypatch.setattr(
        trend_agent,
        "_analyze_trends",
        AsyncMock(return_value={"key_trends": [], "confidence": 0.9}),
    )
    monkeypatch.setattr(
        trend_agent,
        "_generate_recommendations",
        AsyncMock(return_value=[{"type": "product", "action": "introduce"}]),
    )

    result = await trend_agent.process_query(
        "What are the winter trends?",
        {"season": "winter", "location": "cape_town"},
    )

    assert result["status"] == "success"
    assert "analysis" in result
    assert "recommendations" in result


@pytest.mark.asyncio
async def test_process_query_uses_search_and_rag(trend_agent, monkeypatch):
    mock_search = AsyncMock(
        return_value={
            "trends": [{"name": "mock_trend", "relevance": 0.9, "growth": "rising"}]
        }
    )
    mock_rag = AsyncMock(
        return_value=[{"source": "mock_report.pdf", "content": "test"}]
    )
    monkeypatch.setattr(
        trend_agent.trend_searcher, "search_fashion_trends", mock_search
    )
    monkeypatch.setattr(trend_agent.rag_retriever, "retrieve", mock_rag)
    monkeypatch.setattr(
        trend_agent,
        "_get_competitor_insights",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        trend_agent,
        "_analyze_social_trends",
        AsyncMock(return_value={}),
    )

    result = await trend_agent.process_query(
        "What are the key trends for summer in Cape Town?",
        {"season": "summer", "region": "Cape Town", "category": "womenswear"},
    )

    assert result["status"] == "success"
    mock_search.assert_awaited_once()
    mock_rag.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_query_general_error_handling(trend_agent, monkeypatch):
    monkeypatch.setattr(
        trend_agent,
        "_gather_trend_data",
        AsyncMock(side_effect=Exception("Gather failed")),
    )

    result = await trend_agent.process_query("This query is destined to fail.", {})

    assert result["status"] == "error"
    assert "Gather failed" in result["error"]
