"""PostgreSQL data layer with JSON seed fallback."""

from db.session import get_session_factory, init_db

__all__ = ["get_session_factory", "init_db"]
