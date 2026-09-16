"""Tests for unified db service layer."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from db import service as data_service


@pytest.mark.asyncio
async def test_get_product_details_uses_json_fallback(monkeypatch):
    monkeypatch.setenv("USE_JSON_FALLBACK", "true")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    from config.settings import Settings

    monkeypatch.setattr("db.service.settings", Settings.from_env())

    product = await data_service.get_product_details("MF001")
    assert product["product_id"] == "MF001"


@pytest.mark.asyncio
async def test_search_customers_by_name(monkeypatch):
    monkeypatch.setenv("USE_JSON_FALLBACK", "true")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    from config.settings import Settings

    monkeypatch.setattr("db.service.settings", Settings.from_env())

    customers = await data_service.search_customers_by_name("Sarah")
    assert customers
    assert customers[0]["first_name"] == "Sarah"


@pytest.mark.asyncio
async def test_build_knowledge_documents(monkeypatch):
    monkeypatch.setenv("USE_JSON_FALLBACK", "true")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    from config.settings import Settings

    monkeypatch.setattr("db.service.settings", Settings.from_env())

    documents = await data_service.build_knowledge_documents()
    assert len(documents) > 0
    assert documents[0]["content"]
