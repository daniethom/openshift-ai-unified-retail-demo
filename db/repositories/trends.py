"""Fashion trend repository."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import FashionTrend


def trend_to_dict(trend: FashionTrend) -> dict:
    return {
        "trend_id": trend.trend_id,
        "title": trend.title,
        "description": trend.description,
        "season": trend.season,
        "target_demographic": trend.target_demographic,
        "related_categories": trend.related_categories or [],
        "key_colors": trend.key_colors or [],
        "key_materials": trend.key_materials or [],
        "regional_relevance": trend.regional_relevance,
    }


async def get_all(session: AsyncSession) -> list[FashionTrend]:
    result = await session.execute(select(FashionTrend))
    return list(result.scalars().all())
