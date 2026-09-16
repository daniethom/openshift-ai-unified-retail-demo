import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from rag.service import (
    RAGService,
    _format_db_documents,
    build_knowledge_documents,
    fallback_documents,
    get_rag_service,
)


def test_build_knowledge_documents_loads_project_data():
    documents = build_knowledge_documents()

    assert len(documents) > 0
    assert any(doc["doc_type"] == "product" for doc in documents)
    assert any(doc["doc_type"] == "trend" for doc in documents)


def test_format_db_documents_maps_metadata():
    raw = [
        {
            "id": "MF001",
            "source": "products",
            "content": "Product text",
            "metadata": {"product_id": "MF001"},
        }
    ]
    documents = _format_db_documents(raw)

    assert documents[0]["doc_type"] == "product"
    assert documents[0]["source"] == "db/products"


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


def test_rag_service_ensure_collection_creates_when_missing(monkeypatch):
    client = MagicMock()
    client.has_collection.return_value = False

    service = RAGService()
    monkeypatch.setattr(service, "_get_client", lambda: client)

    service.ensure_collection()

    client.create_collection.assert_called_once()


def test_rag_service_retrieve_falls_back_when_collection_missing(monkeypatch):
    monkeypatch.setenv("RAG_USE_FALLBACK", "false")

    import rag.service as rag_service_module
    from config.settings import Settings

    monkeypatch.setattr(rag_service_module, "settings", Settings.from_env())

    client = MagicMock()
    client.has_collection.return_value = False

    service = RAGService()
    monkeypatch.setattr(service, "_get_client", lambda: client)

    docs = service.retrieve_documents("winter coats", top_k=2)

    assert len(docs) == 2


def test_prime_from_data_files_with_mocks(monkeypatch):
    service = RAGService()
    monkeypatch.setattr(service, "ensure_collection", MagicMock())
    monkeypatch.setattr(
        "rag.service.build_knowledge_documents",
        lambda: [
            {
                "id": "1",
                "source": "data/x.json",
                "content": "hello",
                "doc_type": "product",
            }
        ],
    )

    fake_model = MagicMock()
    fake_embedding = MagicMock()
    fake_embedding.tolist.return_value = [0.1] * 384
    fake_model.encode.return_value = [fake_embedding]
    monkeypatch.setattr(service, "_get_model", lambda: fake_model)

    client = MagicMock()
    monkeypatch.setattr(service, "_get_client", lambda: client)

    result = service.prime_from_data_files()

    assert result["inserted"] == 1
    client.insert.assert_called_once()
