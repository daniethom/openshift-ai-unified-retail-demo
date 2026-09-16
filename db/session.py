"""Async SQLAlchemy engine and session factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from config.settings import settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine | None:
    return _engine


def init_db(database_url: str | None = None) -> None:
    """Initialize the async engine when a database URL is configured."""
    global _engine, _session_factory

    url = database_url or settings.database_url
    if not url:
        _engine = None
        _session_factory = None
        return

    _engine = create_async_engine(url, echo=False, pool_pre_ping=True)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


def get_session_factory() -> async_sessionmaker[AsyncSession] | None:
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("Database is not configured. Set DATABASE_URL or USE_JSON_FALLBACK=true.")

    async with _session_factory() as session:
        yield session


init_db()
