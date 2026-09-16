"""SQLAlchemy ORM models mapped from seed JSON files."""

from __future__ import annotations

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    brand: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    sub_category: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    price: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="ZAR")
    stock_level: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    tags: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    first_name: Mapped[str] = mapped_column(String(128), nullable=False)
    last_name: Mapped[str] = mapped_column(String(128), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    phone_number: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    location: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    loyalty_tier: Mapped[str] = mapped_column(String(32), nullable=False, default="Bronze")
    preferred_brands: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    purchase_history: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    demographics: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)


class FashionTrend(Base):
    __tablename__ = "fashion_trends"

    trend_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    season: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    target_demographic: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    related_categories: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    key_colors: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    key_materials: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    regional_relevance: Mapped[str] = mapped_column(Text, nullable=False, default="")


class MarketInsight(Base):
    __tablename__ = "market_insights"

    insight_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    date: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    data_points: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    regional_focus: Mapped[str] = mapped_column(String(128), nullable=False, default="")
