"""
RAG Knowledge Base Module for Meridian Retail AI
Manages the vector database and document storage
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    MilvusException
)
import numpy as np

logger = logging.getLogger(__name__)


class MilvusKnowledgeBase:
    """
    Manages the Milvus vector database for the RAG system
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "meridian_knowledge",
        embedding_dim: int = 384,
        index_type: str = "IVF_FLAT",
        metric_type: str = "IP"  # Inner Product
    ):
        """
        Initialize the Milvus knowledge base
        
        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Name of the collection
            embedding_dim: Dimension of embeddings
            index_type: Type of index to use
            metric_type: Distance metric type
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric_type = metric_type
        self.collection = None
        
        logger.info(f"Initializing MilvusKnowledgeBase: {host}:{port}/{collection_name}")
    
    def connect(self) -> None:
        """Connect to Milvus server"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from Milvus server"""
        connections.disconnect("default")
        logger.info("Disconnected from Milvus")
    
    def create_collection(self, recreate: bool = False) -> None:
        """
        Create the collection with schema
        
        Args:
            recreate: Whether to recreate if exists
        """
        self.connect()
        
        # Check if collection exists
        if utility.has_collection(self.collection_name):
            if recreate:
                logger.info(f"Dropping existing collection: {self.collection_name}")
                Collection(self.collection_name).drop()
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                self.collection = Collection(self.collection_name)
                return
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="timestamp", dtype=DataType.INT64)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Meridian Retail Knowledge Base"
        )
        
        # Create collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            consistency_level="Strong"
        )
        
        logger.info(f"Created collection: {self.collection_name}")
        
        # Create index
        self.create_index()
    
    def create_index(self) -> None:
        """Create index on the embedding field"""
        if self.collection is None:
            raise ValueError("Collection not initialized")
        
        index_params = {
            "metric_type": self.metric_type,
            "index_type": self.index_type,
            "params": {"nlist": 128}
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        logger.info(f"Created index on embedding field with type: {self.index_type}")
    
    def insert_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Insert documents into the collection
        
        Args:
            documents: List of documents with embeddings
            
        Returns:
            List of inserted document IDs
        """
        if not documents:
            return []
        
        if self.collection is None:
            self.create_collection()
        
        # Prepare data for insertion
        ids = []
        contents = []
        embeddings = []
        doc_types = []
        metadatas = []
        timestamps = []
        
        for doc in documents:
            ids.append(doc.get("id", f"doc_{len(ids)}"))
            contents.append(doc.get("content", ""))
            embeddings.append(doc.get("embedding", [0.0] * self.embedding_dim))
            doc_types.append(doc.get("metadata", {}).get("type", "unknown"))
            metadatas.append(json.dumps(doc.get("metadata", {})))
            timestamps.append(int(datetime.now().timestamp()))
        
        # Insert data
        entities = [ids, contents, embeddings, doc_types, metadatas, timestamps]
        
        try:
            self.collection.insert(entities)
            self.collection.flush()
            logger.info(f"Inserted {len(documents)} documents into collection")
            return ids
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            raise
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        doc_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            doc_type: Filter by document type
            filters: Additional filters
            
        Returns:
            List of search results
        """
        if self.collection is None:
            raise ValueError("Collection not initialized")
        
        # Load collection
        self.collection.load()
        
        # Build search parameters
        search_params = {
            "metric_type": self.metric_type,
            "params": {"nprobe": 10}
        }
        
        # Build filter expression
        expr = None
        if doc_type:
            expr = f'doc_type == "{doc_type}"'
        
        # Perform search
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["id", "content", "doc_type", "metadata", "timestamp"]
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "id": hit.entity.get("id"),
                    "content": hit.entity.get("content"),
                    "doc_type": hit.entity.get("doc_type"),
                    "metadata": json.loads(hit.entity.get("metadata", "{}")),
                    "score": hit.score,
                    "timestamp": hit.entity.get("timestamp")
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve documents by IDs
        
        Args:
            ids: List of document IDs
            
        Returns:
            List of documents
        """
        if self.collection is None:
            raise ValueError("Collection not initialized")
        
        self.collection.load()
        
        expr = f'id in {ids}'
        results = self.collection.query(
            expr=expr,
            output_fields=["id", "content", "doc_type", "metadata", "timestamp"]
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.get("id"),
                "content": result.get("content"),
                "doc_type": result.get("doc_type"),
                "metadata": json.loads(result.get("metadata", "{}")),
                "timestamp": result.get("timestamp")
            })
        
        return formatted_results
    
    def delete_by_ids(self, ids: List[str]) -> None:
        """
        Delete documents by IDs
        
        Args:
            ids: List of document IDs to delete
        """
        if self.collection is None:
            raise ValueError("Collection not initialized")
        
        expr = f'id in {ids}'
        self.collection.delete(expr)
        self.collection.flush()
        
        logger.info(f"Deleted {len(ids)} documents")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if self.collection is None:
            return {"status": "not_initialized"}
        
        self.collection.load()
        
        stats = {
            "collection_name": self.collection_name,
            "num_entities": self.collection.num_entities,
            "schema": str(self.collection.schema),
            "index": str(self.collection.indexes)
        }
        
        return stats
    
    def flush(self) -> None:
        """Flush collection data to disk"""
        if self.collection:
            self.collection.flush()
            logger.info("Flushed collection data")
    
    def drop_collection(self) -> None:
        """Drop the entire collection"""
        if utility.has_collection(self.collection_name):
            Collection(self.collection_name).drop()
            logger.info(f"Dropped collection: {self.collection_name}")
            self.collection = None


class DocumentStore:
    """
    Local document storage for the knowledge base
    """
    
    def __init__(self, storage_path: str = "./data/knowledge_base"):
        """
        Initialize document store
        
        Args:
            storage_path: Path to store documents
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Document type paths
        self.paths = {
            "products": self.storage_path / "products",
            "trends": self.storage_path / "trends",
            "insights": self.storage_path / "insights",
            "custom": self.storage_path / "custom"
        }
        
        # Create subdirectories
        for path in self.paths.values():
            path.mkdir(exist_ok=True)
    
    def save_document(
        self,
        doc_id: str,
        content: Dict[str, Any],
        doc_type: str = "custom"
    ) -> str:
        """
        Save a document to disk
        
        Args:
            doc_id: Document ID
            content: Document content
            doc_type: Type of document
            
        Returns:
            Path to saved document
        """
        # Get appropriate path
        base_path = self.paths.get(doc_type, self.paths["custom"])
        
        # Create filename
        filename = f"{doc_id}.json"
        filepath = base_path / filename
        
        # Save document
        with open(filepath, 'w') as f:
            json.dump({
                "id": doc_id,
                "type": doc_type,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Saved document {doc_id} to {filepath}")
        return str(filepath)
    
    def load_document(self, doc_id: str, doc_type: str = None) -> Optional[Dict[str, Any]]:
        """
        Load a document from disk
        
        Args:
            doc_id: Document ID
            doc_type: Type of document (searches all if None)
            
        Returns:
            Document content or None
        """
        # Search paths
        search_paths = [self.paths[doc_type]] if doc_type else self.paths.values()
        
        for path in search_paths:
            filepath = path / f"{doc_id}.json"
            if filepath.exists():
                with open(filepath, 'r') as f:
                    return json.load(f)
        
        return None
    
    def list_documents(self, doc_type: str = None) -> List[Dict[str, Any]]:
        """
        List all documents
        
        Args:
            doc_type: Filter by document type
            
        Returns:
            List of document metadata
        """
        documents = []
        
        # Get paths to search
        search_paths = [self.paths[doc_type]] if doc_type else self.paths.values()
        
        for path in search_paths:
            for filepath in path.glob("*.json"):
                with open(filepath, 'r') as f:
                    doc = json.load(f)
                    documents.append({
                        "id": doc.get("id"),
                        "type": doc.get("type"),
                        "timestamp": doc.get("timestamp"),
                        "path": str(filepath)
                    })
        
        return documents
    
    def delete_document(self, doc_id: str, doc_type: str = None) -> bool:
        """
        Delete a document
        
        Args:
            doc_id: Document ID
            doc_type: Type of document
            
        Returns:
            True if deleted, False if not found
        """
        # Search paths
        search_paths = [self.paths[doc_type]] if doc_type else self.paths.values()
        
        for path in search_paths:
            filepath = path / f"{doc_id}.json"
            if filepath.exists():
                filepath.unlink()
                logger.info(f"Deleted document {doc_id}")
                return True
        
        return False
    
    def import_data_files(self, data_dir: str = "./data") -> Dict[str, int]:
        """
        Import data files from the project data directory
        
        Args:
            data_dir: Path to data directory
            
        Returns:
            Count of imported documents by type
        """
        data_path = Path(data_dir)
        counts = {}
        
        # Import products
        products_file = data_path / "meridian_products.json"
        if products_file.exists():
            with open(products_file, 'r') as f:
                products = json.load(f)
                for product in products:
                    self.save_document(
                        product["product_id"],
                        product,
                        "products"
                    )
                counts["products"] = len(products)
        
        # Import trends
        trends_file = data_path / "fashion_trends.json"
        if trends_file.exists():
            with open(trends_file, 'r') as f:
                trends = json.load(f)
                for trend in trends:
                    self.save_document(
                        trend["trend_id"],
                        trend,
                        "trends"
                    )
                counts["trends"] = len(trends)
        
        # Import market insights
        insights_file = data_path / "sa_market_data.json"
        if insights_file.exists():
            with open(insights_file, 'r') as f:
                insights = json.load(f)
                for insight in insights:
                    self.save_document(
                        insight["insight_id"],
                        insight,
                        "insights"
                    )
                counts["insights"] = len(insights)
        
        logger.info(f"Imported documents: {counts}")
        return counts