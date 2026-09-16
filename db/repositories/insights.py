"""Market insight repository."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import MarketInsight


def insight_to_dict(insight: MarketInsight) -> dict:
    return {
        "insight_id": insight.insight_id,
        "title": insight.title,
        "source": insight.source,
        "date": insight.date,
        "summary": insight.summary,
        "data_points": insight.data_points or [],
        "regional_focus": insight.regional_focus,
    }


async def get_all(session: AsyncSession) -> list[MarketInsight]:
    result = await session.execute(select(MarketInsight))
    return list(result.scalars().all())
