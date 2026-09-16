"""Customer repository."""

from __future__ import annotations

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Customer


def customer_to_dict(customer: Customer) -> dict:
    return {
        "customer_id": customer.customer_id,
        "first_name": customer.first_name,
        "last_name": customer.last_name,
        "email": customer.email,
        "phone_number": customer.phone_number,
        "location": customer.location,
        "loyalty_tier": customer.loyalty_tier,
        "preferred_brands": customer.preferred_brands or [],
        "purchase_history": customer.purchase_history or [],
        "demographics": customer.demographics or {},
    }


async def get_by_id(session: AsyncSession, customer_id: str) -> Customer | None:
    return await session.get(Customer, customer_id)


async def search_by_name(session: AsyncSession, name: str) -> list[Customer]:
    needle = f"%{name.lower()}%"
    result = await session.execute(
        select(Customer).where(
            or_(
                func.lower(Customer.first_name).like(needle),
                func.lower(Customer.last_name).like(needle),
                func.lower(
                    func.concat(Customer.first_name, " ", Customer.last_name)
                ).like(needle),
            )
        )
    )
    return list(result.scalars().all())
