"""Product repository."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Product


def product_to_dict(product: Product) -> dict:
    return {
        "product_id": product.product_id,
        "name": product.name,
        "brand": product.brand,
        "category": product.category,
        "sub_category": product.sub_category,
        "price": product.price,
        "currency": product.currency,
        "stock_level": product.stock_level,
        "description": product.description,
        "tags": product.tags or [],
    }


async def get_all(session: AsyncSession) -> list[Product]:
    result = await session.execute(select(Product))
    return list(result.scalars().all())


async def get_by_id(session: AsyncSession, product_id: str) -> Product | None:
    return await session.get(Product, product_id)


async def count_by_brand(session: AsyncSession, brand_name: str) -> int:
    result = await session.execute(
        select(func.count())
        .select_from(Product)
        .where(func.lower(Product.brand) == brand_name.lower())
    )
    return int(result.scalar_one())


async def total_inventory_value(session: AsyncSession) -> dict:
    result = await session.execute(
        select(
            func.coalesce(func.sum(Product.price * Product.stock_level), 0),
            func.count(Product.product_id),
        )
    )
    total_value, product_count = result.one()
    product_count = int(product_count or 0)
    total_value = float(total_value or 0)
    return {
        "total_stock_value_zar": round(total_value, 2),
        "total_product_count": product_count,
        "average_value_per_product": round(
            total_value / product_count if product_count else 0, 2
        ),
    }
