"""Shared RAG indexing and retrieval backed by Milvus."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient

from config.settings import settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384
DEFAULT_FALLBACK_DOCS = [
    {
        "source": "docs/DEMO_GUIDE.md",
        "content": "The Trend Agent analyzes global and local fashion trends.",
        "score": 0.91,
    },
    {
        "source": "data/fashion_trends.json",
        "content": "Power Tailoring & Suiting: sharply tailored blazers and structured silhouettes.",
        "score": 0.88,
    },
    {
        "source": "data/sa_market_data.json",
        "content": "Cape Town's market shows a preference for quality-focused lifestyle products.",
        "score": 0.85,
    },
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(filename: str) -> list[dict[str, Any]]:
    path = _project_root() / "data" / filename
    if not path.exists():
        logger.warning("Data file not found: %s", path)
        return []
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else [data]


def _product_text(product: dict[str, Any]) -> str:
    tags = ", ".join(product.get("tags", []))
    return (
        f"Product {product.get('name')} ({product.get('product_id')}) "
        f"brand {product.get('brand')} category {product.get('category')}. "
        f"{product.get('description', '')} Tags: {tags}"
    )


def _trend_text(trend: dict[str, Any]) -> str:
    return (
        f"Trend {trend.get('title')} ({trend.get('trend_id')}): "
        f"{trend.get('description', '')} Season: {trend.get('season', '')}. "
        f"Regional relevance: {trend.get('regional_relevance', '')}"
    )


def _insight_text(insight: dict[str, Any]) -> str:
    return (
        f"Market insight {insight.get('title')} ({insight.get('insight_id')}): "
        f"{insight.get('summary', '')} Focus: {insight.get('regional_focus', '')}"
    )


def build_knowledge_documents() -> list[dict[str, Any]]:
    """Build searchable documents from PostgreSQL or seed JSON files."""
    if not settings.use_json_fallback and settings.database_url:
        import asyncio

        from db.service import build_knowledge_documents as db_build

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            raw_docs = asyncio.run(db_build())
            return _format_db_documents(raw_docs)

    return _build_knowledge_documents_from_json()


def _format_db_documents(raw_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for doc in raw_docs:
        metadata = doc.get("metadata", {})
        source = doc.get("source", "postgres")
        doc_type = "product" if source == "products" else source.rstrip("s")
        documents.append(
            {
                "id": doc.get("id", f"doc_{len(documents)}"),
                "source": f"db/{source}",
                "content": doc.get("content", ""),
                "doc_type": doc_type,
                "metadata": metadata,
            }
        )
    return documents


def _build_knowledge_documents_from_json() -> list[dict[str, Any]]:
    """Build searchable documents from project JSON data files."""
    documents: list[dict[str, Any]] = []

    for product in _load_json("meridian_products.json"):
        doc_id = product.get("product_id", f"product_{len(documents)}")
        documents.append(
            {
                "id": str(doc_id),
                "source": "data/meridian_products.json",
                "content": _product_text(product),
                "doc_type": "product",
                "metadata": product,
            }
        )

    for trend in _load_json("fashion_trends.json"):
        doc_id = trend.get("trend_id", f"trend_{len(documents)}")
        documents.append(
            {
                "id": str(doc_id),
                "source": "data/fashion_trends.json",
                "content": _trend_text(trend),
                "doc_type": "trend",
                "metadata": trend,
            }
        )

    for insight in _load_json("sa_market_data.json"):
        doc_id = insight.get("insight_id", f"insight_{len(documents)}")
        documents.append(
            {
                "id": str(doc_id),
                "source": "data/sa_market_data.json",
                "content": _insight_text(insight),
                "doc_type": "insight",
                "metadata": insight,
            }
        )

    return documents


def fallback_documents(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Return static fallback documents when Milvus is unavailable."""
    query_lower = query.lower()
    ranked = sorted(
        DEFAULT_FALLBACK_DOCS,
        key=lambda doc: int(
            any(token in doc["content"].lower() for token in query_lower.split())
        ),
        reverse=True,
    )
    return ranked[:top_k]


class RAGService:
    """Milvus-backed retrieval service with graceful fallback."""

    def __init__(self) -> None:
        self._client: MilvusClient | None = None
        self._model = None

    @property
    def collection_name(self) -> str:
        return settings.milvus_collection_name

    def _get_client(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(uri=settings.milvus_uri)
        return self._client

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(settings.embedding_model)
        return self._model

    def ensure_collection(self, recreate: bool = False) -> None:
        client = self._get_client()
        if client.has_collection(self.collection_name):
            if recreate:
                client.drop_collection(self.collection_name)
            else:
                return

        client.create_collection(
            collection_name=self.collection_name,
            dimension=EMBEDDING_DIM,
            metric_type="IP",
        )
        logger.info("Created Milvus collection %s", self.collection_name)

    def prime_from_data_files(self, recreate: bool = False) -> dict[str, int]:
        """Load JSON data files and insert embeddings into Milvus."""
        documents = build_knowledge_documents()
        if not documents:
            return {"inserted": 0}

        self.ensure_collection(recreate=recreate)
        model = self._get_model()
        texts = [doc["content"] for doc in documents]
        embeddings = model.encode(texts, convert_to_numpy=True)

        rows = []
        for doc, embedding in zip(documents, embeddings, strict=True):
            rows.append(
                {
                    "vector": embedding.tolist(),
                    "source": doc["source"],
                    "content": doc["content"],
                    "doc_type": doc["doc_type"],
                    "doc_id": doc["id"],
                }
            )

        client = self._get_client()
        client.insert(collection_name=self.collection_name, data=rows)
        logger.info("Inserted %s documents into %s", len(rows), self.collection_name)
        return {"inserted": len(rows)}

    def retrieve_documents(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve the most relevant documents for a query."""
        if settings.rag_use_fallback:
            return fallback_documents(query, top_k)

        try:
            client = self._get_client()
            if not client.has_collection(self.collection_name):
                logger.warning("Milvus collection %s not found", self.collection_name)
                return fallback_documents(query, top_k)

            model = self._get_model()
            embedding = model.encode([query], convert_to_numpy=True)[0].tolist()
            results = client.search(
                collection_name=self.collection_name,
                data=[embedding],
                limit=top_k,
                output_fields=["source", "content", "doc_type"],
            )

            documents: list[dict[str, Any]] = []
            for hits in results:
                for hit in hits:
                    entity = hit.get("entity", {})
                    score = hit.get("distance", hit.get("score", 0.0))
                    documents.append(
                        {
                            "source": entity.get("source", "unknown"),
                            "content": entity.get("content", ""),
                            "score": float(score),
                            "doc_type": entity.get("doc_type", "unknown"),
                        }
                    )

            return documents or fallback_documents(query, top_k)
        except Exception as exc:
            logger.warning("Milvus retrieval failed, using fallback: %s", exc)
            return fallback_documents(query, top_k)


_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
