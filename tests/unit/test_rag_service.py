import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from rag.service import build_knowledge_documents, fallback_documents, get_rag_service


def test_build_knowledge_documents_loads_project_data():
    documents = build_knowledge_documents()

    assert len(documents) > 0
    assert any(doc["doc_type"] == "product" for doc in documents)
    assert any(doc["doc_type"] == "trend" for doc in documents)


def test_fallback_documents_returns_results():
    docs = fallback_documents("winter fashion Cape Town", top_k=2)

    assert len(docs) == 2
    assert "source" in docs[0]
    assert "content" in docs[0]


def test_retrieve_documents_uses_fallback_when_enabled(monkeypatch):
    monkeypatch.setenv("RAG_USE_FALLBACK", "true")

    import rag.service as rag_service_module
    from config.settings import Settings

    monkeypatch.setattr(rag_service_module, "settings", Settings.from_env())
    monkeypatch.setattr(rag_service_module, "_rag_service", None)

    service = get_rag_service()
    docs = service.retrieve_documents("professional women winter trends", top_k=3)

    assert len(docs) == 3
