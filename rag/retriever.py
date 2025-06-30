"""
RAG Retriever Module for Meridian Retail AI
Implements the retrieval logic for the RAG system
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import json

from .embeddings import EmbeddingGenerator, DocumentProcessor
from .knowledge_base import MilvusKnowledgeBase, DocumentStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Main retriever class for the RAG system
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        collection_name: str = "meridian_knowledge",
        storage_path: str = "./data/knowledge_base"
    ):
        """
        Initialize the RAG retriever
        
        Args:
            embedding_model: Name of the sentence transformer model
            milvus_host: Milvus server host
            milvus_port: Milvus server port
            collection_name: Name of the Milvus collection
            storage_path: Path for local document storage
        """
        # Initialize components
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.document_processor = DocumentProcessor(self.embedding_generator)
        self.knowledge_base = MilvusKnowledgeBase(
            host=milvus_host,
            port=milvus_port,
            collection_name=collection_name,
            embedding_dim=384  # Default for all-MiniLM-L6-v2
        )
        self.document_store = DocumentStore(storage_path)
        
        # Initialize embedding model
        self.embedding_generator.load_model()
        
        logger.info("RAGRetriever initialized successfully")
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            doc_type: Filter by document type
            filters: Additional filters
            rerank: Whether to rerank results
            
        Returns:
            List of retrieved documents with scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single_embedding(query)
            
            # Search in vector database
            results = self.knowledge_base.search(
                query_embedding=query_embedding,
                top_k=top_k * 2 if rerank else top_k,  # Get more for reranking
                doc_type=doc_type,
                filters=filters
            )
            
            # Rerank if requested
            if rerank and len(results) > top_k:
                results = self._rerank_results(query, results, top_k)
            
            # Enhance results with additional context
            enhanced_results = self._enhance_results(results)
            
            logger.info(f"Retrieved {len(enhanced_results)} documents for query: {query[:50]}...")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank results based on additional criteria
        
        Args:
            query: Original query
            results: Initial search results
            top_k: Number of results to return
            
        Returns:
            Reranked results
        """
        # Simple reranking based on metadata relevance
        query_lower = query.lower()
        
        for result in results:
            boost_score = 0.0
            
            # Boost by type relevance
            if "trend" in query_lower and result.get("doc_type") == "trend":
                boost_score += 0.2
            elif "product" in query_lower and result.get("doc_type") == "product":
                boost_score += 0.2
            elif "market" in query_lower and result.get("doc_type") == "market_insight":
                boost_score += 0.2
            
            # Boost by keyword matches in metadata
            metadata = result.get("metadata", {})
            if any(keyword in query_lower for keyword in ["winter", "cold", "coat"]):
                if metadata.get("season") == "Winter 2025":
                    boost_score += 0.15
            
            # Boost by recency (if timestamp available)
            if result.get("timestamp"):
                days_old = (datetime.now().timestamp() - result["timestamp"]) / 86400
                if days_old < 30:
                    boost_score += 0.1
                elif days_old < 90:
                    boost_score += 0.05
            
            # Apply boost
            result["rerank_score"] = result.get("score", 0) + boost_score
        
        # Sort by rerank score
        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        return results[:top_k]
    
    def _enhance_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enhance results with additional context
        
        Args:
            results: Search results
            
        Returns:
            Enhanced results
        """
        enhanced = []
        
        for result in results:
            enhanced_result = result.copy()
            
            # Add source information
            metadata = result.get("metadata", {})
            doc_type = result.get("doc_type", "unknown")
            
            if doc_type == "product":
                enhanced_result["source"] = f"Product Catalog - {metadata.get('brand', 'Unknown')}"
                enhanced_result["display_title"] = metadata.get("name", "Unknown Product")
            elif doc_type == "trend":
                enhanced_result["source"] = f"Trend Analysis - {metadata.get('season', 'Unknown')}"
                enhanced_result["display_title"] = metadata.get("title", "Unknown Trend")
            elif doc_type == "market_insight":
                enhanced_result["source"] = f"Market Research - {metadata.get('source', 'Unknown')}"
                enhanced_result["display_title"] = metadata.get("title", "Unknown Insight")
            else:
                enhanced_result["source"] = "Knowledge Base"
                enhanced_result["display_title"] = f"Document {result.get('id', 'Unknown')}"
            
            # Format content preview
            content = result.get("content", "")
            enhanced_result["preview"] = content[:200] + "..." if len(content) > 200 else content
            
            enhanced.append(enhanced_result)
        
        return enhanced
    
    async def index_documents(
        self,
        documents: List[Dict[str, Any]],
        doc_type: str = "custom",
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Index new documents into the knowledge base
        
        Args:
            documents: List of documents to index
            doc_type: Type of documents
            batch_size: Batch size for processing
            
        Returns:
            Indexing statistics
        """
        if not documents:
            return {"indexed": 0, "failed": 0, "errors": []}
        
        stats = {"indexed": 0, "failed": 0, "errors": []}
        
        try:
            # Process documents based on type
            if doc_type == "product":
                processed = self.document_processor.process_product_data(documents)
            elif doc_type == "trend":
                processed = self.document_processor.process_trend_data(documents)
            elif doc_type == "market_insight":
                processed = self.document_processor.process_market_insights(documents)
            else:
                # Generic processing
                processed = []
                for doc in documents:
                    processed_doc = {
                        "id": doc.get("id", f"doc_{len(processed)}"),
                        "content": doc.get("content", ""),
                        "metadata": {
                            "type": doc_type,
                            **doc.get("metadata", {})
                        }
                    }
                    processed.append(processed_doc)
                
                # Add embeddings
                processed = self.embedding_generator.batch_process_documents(
                    processed, batch_size=batch_size
                )
            
            # Insert into Milvus
            inserted_ids = self.knowledge_base.insert_documents(processed)
            stats["indexed"] = len(inserted_ids)
            
            # Save to document store
            for doc in processed:
                self.document_store.save_document(
                    doc_id=doc["id"],
                    content=doc,
                    doc_type=doc_type
                )
            
            logger.info(f"Successfully indexed {stats['indexed']} {doc_type} documents")
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            stats["failed"] = len(documents)
            stats["errors"].append(str(e))
        
        return stats
    
    async def update_document(
        self,
        doc_id: str,
        content: Dict[str, Any],
        doc_type: str = "custom"
    ) -> bool:
        """
        Update an existing document
        
        Args:
            doc_id: Document ID
            content: New content
            doc_type: Document type
            
        Returns:
            Success status
        """
        try:
            # Delete old version
            self.knowledge_base.delete_by_ids([doc_id])
            self.document_store.delete_document(doc_id, doc_type)
            
            # Index new version
            stats = await self.index_documents([content], doc_type)
            
            return stats["indexed"] > 0
            
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False
    
    async def delete_documents(
        self,
        doc_ids: List[str],
        doc_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Delete documents from the knowledge base
        
        Args:
            doc_ids: List of document IDs
            doc_type: Document type (for local storage)
            
        Returns:
            Deletion statistics
        """
        stats = {"deleted": 0, "failed": 0, "errors": []}
        
        try:
            # Delete from Milvus
            self.knowledge_base.delete_by_ids(doc_ids)
            
            # Delete from document store
            for doc_id in doc_ids:
                if self.document_store.delete_document(doc_id, doc_type):
                    stats["deleted"] += 1
                else:
                    stats["failed"] += 1
            
            logger.info(f"Deleted {stats['deleted']} documents")
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            stats["failed"] = len(doc_ids)
            stats["errors"].append(str(e))
        
        return stats
    
    async def search_by_metadata(
        self,
        filters: Dict[str, Any],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search documents by metadata filters
        
        Args:
            filters: Metadata filters
            top_k: Number of results
            
        Returns:
            Matching documents
        """
        # This is a simplified implementation
        # In production, you would build proper Milvus expressions
        all_docs = self.document_store.list_documents()
        matching = []
        
        for doc_meta in all_docs:
            doc = self.document_store.load_document(doc_meta["id"])
            if doc:
                metadata = doc.get("content", {}).get("metadata", {})
                
                # Check if all filters match
                match = True
                for key, value in filters.items():
                    if metadata.get(key) != value:
                        match = False
                        break
                
                if match:
                    matching.append(doc["content"])
                    if len(matching) >= top_k:
                        break
        
        return matching
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        try:
            # Get Milvus stats
            milvus_stats = self.knowledge_base.get_collection_stats()
            
            # Get document store stats
            doc_counts = {}
            for doc_type in ["products", "trends", "insights", "custom"]:
                docs = self.document_store.list_documents(doc_type)
                doc_counts[doc_type] = len(docs)
            
            return {
                "milvus": milvus_stats,
                "documents": doc_counts,
                "total_documents": sum(doc_counts.values()),
                "embedding_model": self.embedding_generator.get_model_info()
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
    
    async def initialize_from_data_files(self) -> Dict[str, Any]:
        """Initialize knowledge base from project data files"""
        try:
            # Import data files to document store
            import_counts = self.document_store.import_data_files()
            
            # Index each type
            total_stats = {"products": {}, "trends": {}, "insights": {}}
            
            # Index products
            products = []
            for doc_meta in self.document_store.list_documents("products"):
                doc = self.document_store.load_document(doc_meta["id"])
                if doc:
                    products.append(doc["content"])
            
            if products:
                total_stats["products"] = await self.index_documents(
                    products, doc_type="product"
                )
            
            # Index trends
            trends = []
            for doc_meta in self.document_store.list_documents("trends"):
                doc = self.document_store.load_document(doc_meta["id"])
                if doc:
                    trends.append(doc["content"])
            
            if trends:
                total_stats["trends"] = await self.index_documents(
                    trends, doc_type="trend"
                )
            
            # Index insights
            insights = []
            for doc_meta in self.document_store.list_documents("insights"):
                doc = self.document_store.load_document(doc_meta["id"])
                if doc:
                    insights.append(doc["content"])
            
            if insights:
                total_stats["insights"] = await self.index_documents(
                    insights, doc_type="market_insight"
                )
            
            return {
                "imported": import_counts,
                "indexed": total_stats
            }
            
        except Exception as e:
            logger.error(f"Error initializing from data files: {e}")
            return {"error": str(e)}


class HybridRetriever(RAGRetriever):
    """
    Hybrid retriever that combines vector search with keyword search
    """
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using both vector and keyword search
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            doc_type: Filter by document type
            filters: Additional filters
            vector_weight: Weight for vector search results
            keyword_weight: Weight for keyword search results
            
        Returns:
            Combined and reranked results
        """
        # Get vector search results
        vector_results = await super().retrieve(
            query=query,
            top_k=top_k * 2,
            doc_type=doc_type,
            filters=filters,
            rerank=False
        )
        
        # Get keyword search results
        keyword_results = await self._keyword_search(
            query=query,
            top_k=top_k * 2,
            doc_type=doc_type
        )
        
        # Combine and rerank
        combined_results = self._combine_results(
            vector_results,
            keyword_results,
            vector_weight,
            keyword_weight
        )
        
        return combined_results[:top_k]
    
    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        doc_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search
        
        Args:
            query: Query string
            top_k: Number of results
            doc_type: Document type filter
            
        Returns:
            Keyword search results
        """
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        results = []
        all_docs = self.document_store.list_documents(doc_type)
        
        for doc_meta in all_docs:
            doc = self.document_store.load_document(doc_meta["id"])
            if doc:
                content = doc.get("content", {})
                text = content.get("content", "").lower()
                
                # Calculate keyword score
                score = 0.0
                for term in query_terms:
                    score += text.count(term) / len(text.split())
                
                if score > 0:
                    results.append({
                        **content,
                        "score": score,
                        "search_type": "keyword"
                    })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def _combine_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        vector_weight: float,
        keyword_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Combine and rerank results from different search methods
        
        Args:
            vector_results: Vector search results
            keyword_results: Keyword search results
            vector_weight: Weight for vector results
            keyword_weight: Weight for keyword results
            
        Returns:
            Combined results
        """
        # Normalize scores
        max_vector_score = max([r.get("score", 0) for r in vector_results], default=1)
        max_keyword_score = max([r.get("score", 0) for r in keyword_results], default=1)
        
        # Create combined scores
        combined = {}
        
        for result in vector_results:
            doc_id = result.get("id")
            normalized_score = result.get("score", 0) / max_vector_score
            combined[doc_id] = {
                **result,
                "vector_score": normalized_score,
                "keyword_score": 0,
                "combined_score": normalized_score * vector_weight
            }
        
        for result in keyword_results:
            doc_id = result.get("id")
            normalized_score = result.get("score", 0) / max_keyword_score
            
            if doc_id in combined:
                combined[doc_id]["keyword_score"] = normalized_score
                combined[doc_id]["combined_score"] += normalized_score * keyword_weight
            else:
                combined[doc_id] = {
                    **result,
                    "vector_score": 0,
                    "keyword_score": normalized_score,
                    "combined_score": normalized_score * keyword_weight
                }
        
        # Sort by combined score
        final_results = list(combined.values())
        final_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return final_results