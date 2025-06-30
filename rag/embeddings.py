"""
RAG Embeddings Module for Meridian Retail AI
Handles text embedding generation for the vector database
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for text documents using sentence transformers
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing EmbeddingGenerator with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
    def load_model(self) -> None:
        """Load the embedding model"""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
            logger.info("Model loaded successfully")
    
    def generate_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            
        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        # Ensure model is loaded
        self.load_model()
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string to embed
            
        Returns:
            numpy array embedding
        """
        return self.generate_embeddings([text])[0]
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        # Basic preprocessing
        text = text.strip()
        text = " ".join(text.split())  # Normalize whitespace
        
        # Limit length to avoid memory issues
        max_length = 512  # Token limit for most models
        if len(text) > max_length * 4:  # Rough char to token ratio
            text = text[:max_length * 4]
            logger.warning(f"Text truncated to {len(text)} characters")
        
        return text
    
    def batch_process_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "content",
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of documents and add embeddings
        
        Args:
            documents: List of document dictionaries
            text_field: Field name containing the text to embed
            batch_size: Batch size for processing
            
        Returns:
            Documents with added embeddings
        """
        if not documents:
            return []
        
        # Extract texts
        texts = []
        for doc in documents:
            text = doc.get(text_field, "")
            preprocessed = self.preprocess_text(text)
            texts.append(preprocessed)
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts, batch_size)
        
        # Add embeddings to documents
        for i, doc in enumerate(documents):
            doc["embedding"] = embeddings[i].tolist()
            doc["embedding_model"] = self.model_name
        
        return documents
    
    def calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        # Ensure numpy arrays
        if isinstance(embedding1, list):
            embedding1 = np.array(embedding1)
        if isinstance(embedding2, list):
            embedding2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def find_similar(
        self,
        query_embedding: np.ndarray,
        embeddings: List[np.ndarray],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[int, float]]:
        """
        Find most similar embeddings to a query
        
        Args:
            query_embedding: Query embedding
            embeddings: List of embeddings to search
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if not embeddings:
            return []
        
        # Calculate similarities
        similarities = []
        for i, emb in enumerate(embeddings):
            sim = self.calculate_similarity(query_embedding, emb)
            if sim >= threshold:
                similarities.append((i, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        return similarities[:top_k]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        self.load_model()
        
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "max_seq_length": self.model.max_seq_length,
            "device": self.device,
            "model_loaded": self.model is not None
        }


class DocumentProcessor:
    """
    Processes documents for the RAG system
    """
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator
        
    def process_product_data(
        self,
        products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process product data for embedding
        
        Args:
            products: List of product dictionaries
            
        Returns:
            Processed products with embeddings
        """
        processed = []
        
        for product in products:
            # Create searchable text
            text_parts = [
                f"Product: {product.get('name', '')}",
                f"Brand: {product.get('brand', '')}",
                f"Category: {product.get('category', '')} - {product.get('sub_category', '')}",
                f"Description: {product.get('description', '')}",
                f"Tags: {', '.join(product.get('tags', []))}"
            ]
            
            searchable_text = " ".join(text_parts)
            
            # Create document
            doc = {
                "id": product.get("product_id"),
                "content": searchable_text,
                "metadata": {
                    "type": "product",
                    "product_id": product.get("product_id"),
                    "name": product.get("name"),
                    "brand": product.get("brand"),
                    "category": product.get("category"),
                    "price": product.get("price"),
                    "stock_level": product.get("stock_level")
                }
            }
            
            processed.append(doc)
        
        # Add embeddings
        return self.embedding_generator.batch_process_documents(processed)
    
    def process_trend_data(
        self,
        trends: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process fashion trend data for embedding
        
        Args:
            trends: List of trend dictionaries
            
        Returns:
            Processed trends with embeddings
        """
        processed = []
        
        for trend in trends:
            # Create searchable text
            text_parts = [
                f"Trend: {trend.get('title', '')}",
                f"Description: {trend.get('description', '')}",
                f"Season: {trend.get('season', '')}",
                f"Target: {trend.get('target_demographic', '')}",
                f"Categories: {', '.join(trend.get('related_categories', []))}",
                f"Colors: {', '.join(trend.get('key_colors', []))}",
                f"Materials: {', '.join(trend.get('key_materials', []))}",
                f"Regional relevance: {trend.get('regional_relevance', '')}"
            ]
            
            searchable_text = " ".join(text_parts)
            
            # Create document
            doc = {
                "id": trend.get("trend_id"),
                "content": searchable_text,
                "metadata": {
                    "type": "trend",
                    "trend_id": trend.get("trend_id"),
                    "title": trend.get("title"),
                    "season": trend.get("season"),
                    "target_demographic": trend.get("target_demographic")
                }
            }
            
            processed.append(doc)
        
        # Add embeddings
        return self.embedding_generator.batch_process_documents(processed)
    
    def process_market_insights(
        self,
        insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process market insights data for embedding
        
        Args:
            insights: List of insight dictionaries
            
        Returns:
            Processed insights with embeddings
        """
        processed = []
        
        for insight in insights:
            # Create searchable text
            text_parts = [
                f"Insight: {insight.get('title', '')}",
                f"Source: {insight.get('source', '')}",
                f"Summary: {insight.get('summary', '')}",
                f"Region: {insight.get('regional_focus', '')}",
            ]
            
            # Add data points
            for dp in insight.get("data_points", []):
                text_parts.append(f"{dp.get('metric', '')}: {dp.get('value', '')}")
            
            searchable_text = " ".join(text_parts)
            
            # Create document
            doc = {
                "id": insight.get("insight_id"),
                "content": searchable_text,
                "metadata": {
                    "type": "market_insight",
                    "insight_id": insight.get("insight_id"),
                    "title": insight.get("title"),
                    "source": insight.get("source"),
                    "date": insight.get("date"),
                    "regional_focus": insight.get("regional_focus")
                }
            }
            
            processed.append(doc)
        
        # Add embeddings
        return self.embedding_generator.batch_process_documents(processed)
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 128
    ) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end
                for sep in [". ", "! ", "? ", "\n\n", "\n"]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep != -1:
                        end = last_sep + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks