"""Load data/*.json into PostgreSQL idempotently."""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy.dialects.postgresql import insert

from config.settings import settings
from db import json_store
from db.models import Base, Customer, FashionTrend, MarketInsight, Product
from db.session import get_engine, init_db, session_scope

logger = logging.getLogger(__name__)


async def _upsert_products() -> int:
    rows = json_store.load_products()
    async with session_scope() as session:
        for row in rows:
            stmt = insert(Product).values(
                product_id=row["product_id"],
                name=row.get("name", ""),
                brand=row.get("brand", ""),
                category=row.get("category", ""),
                sub_category=row.get("sub_category", ""),
                price=float(row.get("price", 0)),
                currency=row.get("currency", "ZAR"),
                stock_level=int(row.get("stock_level", 0)),
                description=row.get("description", ""),
                tags=row.get("tags", []),
            ).on_conflict_do_update(
                index_elements=[Product.product_id],
                set_={
                    "name": row.get("name", ""),
                    "brand": row.get("brand", ""),
                    "category": row.get("category", ""),
                    "sub_category": row.get("sub_category", ""),
                    "price": float(row.get("price", 0)),
                    "currency": row.get("currency", "ZAR"),
                    "stock_level": int(row.get("stock_level", 0)),
                    "description": row.get("description", ""),
                    "tags": row.get("tags", []),
                },
            )
            await session.execute(stmt)
        await session.commit()
    return len(rows)


async def _upsert_customers() -> int:
    rows = json_store.load_customers()
    async with session_scope() as session:
        for row in rows:
            stmt = insert(Customer).values(
                customer_id=row["customer_id"],
                first_name=row.get("first_name", ""),
                last_name=row.get("last_name", ""),
                email=row.get("email", ""),
                phone_number=row.get("phone_number", ""),
                location=row.get("location", ""),
                loyalty_tier=row.get("loyalty_tier", "Bronze"),
                preferred_brands=row.get("preferred_brands", []),
                purchase_history=row.get("purchase_history", []),
                demographics=row.get("demographics", {}),
            ).on_conflict_do_update(
                index_elements=[Customer.customer_id],
                set_={
                    "first_name": row.get("first_name", ""),
                    "last_name": row.get("last_name", ""),
                    "email": row.get("email", ""),
                    "phone_number": row.get("phone_number", ""),
                    "location": row.get("location", ""),
                    "loyalty_tier": row.get("loyalty_tier", "Bronze"),
                    "preferred_brands": row.get("preferred_brands", []),
                    "purchase_history": row.get("purchase_history", []),
                    "demographics": row.get("demographics", {}),
                },
            )
            await session.execute(stmt)
        await session.commit()
    return len(rows)


async def _upsert_trends() -> int:
    rows = json_store.load_trends()
    async with session_scope() as session:
        for row in rows:
            stmt = insert(FashionTrend).values(
                trend_id=row["trend_id"],
                title=row.get("title", ""),
                description=row.get("description", ""),
                season=row.get("season", ""),
                target_demographic=row.get("target_demographic", ""),
                related_categories=row.get("related_categories", []),
                key_colors=row.get("key_colors", []),
                key_materials=row.get("key_materials", []),
                regional_relevance=row.get("regional_relevance", ""),
            ).on_conflict_do_update(
                index_elements=[FashionTrend.trend_id],
                set_={
                    "title": row.get("title", ""),
                    "description": row.get("description", ""),
                    "season": row.get("season", ""),
                    "target_demographic": row.get("target_demographic", ""),
                    "related_categories": row.get("related_categories", []),
                    "key_colors": row.get("key_colors", []),
                    "key_materials": row.get("key_materials", []),
                    "regional_relevance": row.get("regional_relevance", ""),
                },
            )
            await session.execute(stmt)
        await session.commit()
    return len(rows)


async def _upsert_insights() -> int:
    rows = json_store.load_insights()
    async with session_scope() as session:
        for row in rows:
            stmt = insert(MarketInsight).values(
                insight_id=row["insight_id"],
                title=row.get("title", ""),
                source=row.get("source", ""),
                date=row.get("date", ""),
                summary=row.get("summary", ""),
                data_points=row.get("data_points", []),
                regional_focus=row.get("regional_focus", ""),
            ).on_conflict_do_update(
                index_elements=[MarketInsight.insight_id],
                set_={
                    "title": row.get("title", ""),
                    "source": row.get("source", ""),
                    "date": row.get("date", ""),
                    "summary": row.get("summary", ""),
                    "data_points": row.get("data_points", []),
                    "regional_focus": row.get("regional_focus", ""),
                },
            )
            await session.execute(stmt)
        await session.commit()
    return len(rows)


async def seed_all() -> dict[str, int]:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL must be set to seed PostgreSQL.")

    init_db(settings.database_url)
    counts = {
        "products": await _upsert_products(),
        "customers": await _upsert_customers(),
        "fashion_trends": await _upsert_trends(),
        "market_insights": await _upsert_insights(),
    }
    logger.info("Seeded database: %s", counts)
    return counts


async def create_schema() -> None:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL must be set to create schema.")

    init_db(settings.database_url)
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Failed to initialize database engine.")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def main() -> None:
    logging.basicConfig(level=settings.log_level)
    asyncio.run(seed_all())


if __name__ == "__main__":
    main()
