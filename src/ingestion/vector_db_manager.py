"""
Vector database management for storing and retrieving document embeddings.
"""
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import faiss
from loguru import logger

from ..models.data_models import DocumentChunk, RetrievalResult
from ..utils.config_manager import ConfigManager


class VectorDBInterface(ABC):
    """Abstract interface for vector databases."""
    
    @abstractmethod
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector database."""
        print("--Interface add docs--")
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        pass


class ChromaDBManager(VectorDBInterface):
    """ChromaDB implementation for vector storage."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.persist_directory = config.get("vector_db.persist_directory", "./data/vector_db")
        self.collection_name = config.get("vector_db.collection_name", "documents")
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized ChromaDB with collection: {self.collection_name}")
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to ChromaDB."""
        print("--Chroma add docs--")
        try:
            if not chunks:
                logger.warning("No chunks provided for indexing")
                return
            
            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in chunks]
            embeddings = [chunk.embedding for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]

            
            # Add to collection
            BATCH_SIZE = 300     # or whatever your system can handle
            total = len(chunks)
            for i in range(0, total, BATCH_SIZE):
                print(f"Batch {i}")
                batch = chunks[i : i + BATCH_SIZE]
                ids        = [c.chunk_id   for c in batch]
                embeddings = [c.embedding  for c in batch]
                documents  = [c.content    for c in batch]
                metadatas  = [c.metadata   for c in batch]

                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                logger.info(f"✅ Indexed batch {i}-{min(i+BATCH_SIZE, total)} / {total}")
            
            logger.info(f"Added {len(chunks)} chunks to ChromaDB")
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Search for similar documents in ChromaDB."""
        try:
            # Generate embedding for query
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(
                self.config.get("embeddings.model_name", "sentence-transformers/all-MiniLM-L6-v2")
            )
            query_embedding = embedding_model.encode([query]).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            if results['ids'] and results['ids'][0]:
                for i, chunk_id in enumerate(results['ids'][0]):
                    retrieval_results.append(RetrievalResult(
                        chunk_id=chunk_id,
                        content=results['documents'][0][i],
                        similarity_score=1.0 - results['distances'][0][i],  # Convert distance to similarity
                        metadata=results['metadatas'][0][i],
                        source_document=results['metadatas'][0][i].get('source_file', 'Unknown')
                    ))
            
            logger.info(f"Found {len(retrieval_results)} similar documents")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error searching in ChromaDB: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection."""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise


class FAISSManager(VectorDBInterface):
    """FAISS implementation for vector storage."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.persist_directory = config.get("vector_db.persist_directory", "./data/vector_db")
        self.collection_name = config.get("vector_db.collection_name", "documents")
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize FAISS index
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        self.index = None
        self.chunks_metadata = []
        
        # File paths
        self.index_file = os.path.join(self.persist_directory, f"{self.collection_name}.index")
        self.metadata_file = os.path.join(self.persist_directory, f"{self.collection_name}_metadata.json")
        
        # Load existing index if available
        self._load_index()
        
        logger.info(f"Initialized FAISS with collection: {self.collection_name}")
    
    def _load_index(self):
        """Load existing FAISS index and metadata."""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                self.index = faiss.read_index(self.index_file)
                
                with open(self.metadata_file, 'r') as f:
                    self.chunks_metadata = json.load(f)
                
                logger.info(f"Loaded existing FAISS index with {len(self.chunks_metadata)} documents")
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
                logger.info("Created new FAISS index")
                
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            self.index = faiss.IndexFlatIP(self.dimension)
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            faiss.write_index(self.index, self.index_file)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self.chunks_metadata, f, indent=2)
            
            logger.info("Saved FAISS index and metadata")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
            raise
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to FAISS index."""
        try:
            if not chunks:
                logger.warning("No chunks provided for indexing")
                return
            
            # Prepare embeddings and metadata
            embeddings = np.array([chunk.embedding for chunk in chunks], dtype=np.float32)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.index.add(embeddings)
            
            # Store metadata
            for chunk in chunks:
                self.chunks_metadata.append({
                    'chunk_id': chunk.chunk_id,
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'source_document': chunk.metadata.get('source_file', 'Unknown')
                })
            
            # Save to disk
            self._save_index()
            
            logger.info(f"Added {len(chunks)} chunks to FAISS index")
            
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Search for similar documents in FAISS index."""
        try:
            if self.index.ntotal == 0:
                logger.warning("FAISS index is empty")
                return []
            
            # Generate embedding for query
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(
                self.config.get("embeddings.model_name", "sentence-transformers/all-MiniLM-L6-v2")
            )
            query_embedding = embedding_model.encode([query]).astype(np.float32)
            
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS
            k = min(k, self.index.ntotal)
            scores, indices = self.index.search(query_embedding, k)
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # Valid result
                    metadata = self.chunks_metadata[idx]
                    retrieval_results.append(RetrievalResult(
                        chunk_id=metadata['chunk_id'],
                        content=metadata['content'],
                        similarity_score=float(scores[0][i]),
                        metadata=metadata['metadata'],
                        source_document=metadata['source_document']
                    ))
            
            logger.info(f"Found {len(retrieval_results)} similar documents")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error searching in FAISS: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the FAISS collection."""
        try:
            return {
                "total_documents": self.index.ntotal if self.index else 0,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "dimension": self.dimension
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}


class VectorDBFactory:
    """Factory for creating vector database instances."""
    
    @staticmethod
    def create_vector_db(config: ConfigManager) -> VectorDBInterface:
        """Create vector database instance based on configuration."""
        db_type = config.get("vector_db.type", "chroma").lower()
        
        if db_type == "chroma":
            return ChromaDBManager(config)
        elif db_type == "faiss":
            return FAISSManager(config)
        else:
            raise ValueError(f"Unsupported vector database type: {db_type}")


class HybridRetriever:
    """Hybrid retriever combining multiple search strategies."""
    
    def __init__(self, vector_db: VectorDBInterface, config: ConfigManager):
        self.vector_db = vector_db
        self.config = config
        self.similarity_threshold = config.get("retrieval.similarity_threshold", 0.3)
    
    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Retrieve relevant documents using hybrid approach."""
        try:
            # Get initial results from vector search
            results = self.vector_db.similarity_search(query, k)
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result.similarity_score >= self.similarity_threshold
            ]
            
            # Apply reranking if enabled
            if self.config.get("retrieval.rerank", True):
                filtered_results = self._rerank_results(query, filtered_results)
            
            # Limit to max context length
            max_context_length = self.config.get("retrieval.max_context_length", 4000)
            final_results = self._limit_context_length(filtered_results, max_context_length)
            
            logger.info(f"Retrieved {len(final_results)} relevant documents")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            raise
    
    def _rerank_results(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank results based on additional criteria."""
        # Simple reranking based on content length and keyword matching
        for result in results:
            # Boost score for keyword matches
            query_terms = query.lower().split()
            content_lower = result.content.lower()
            
            keyword_matches = sum(1 for term in query_terms if term in content_lower)
            keyword_boost = keyword_matches / len(query_terms) * 0.1
            
            # Penalize very short or very long chunks
            content_length = len(result.content)
            if content_length < 100:
                length_penalty = 0.1
            elif content_length > 2000:
                length_penalty = 0.05
            else:
                length_penalty = 0
            
            result.similarity_score = min(1.0, result.similarity_score + keyword_boost - length_penalty)
        
        # Sort by adjusted similarity score
        return sorted(results, key=lambda x: x.similarity_score, reverse=True)
    
    def _limit_context_length(self, results: List[RetrievalResult], max_length: int) -> List[RetrievalResult]:
        """Limit results to fit within context length."""
        current_length = 0
        limited_results = []
        
        for result in results:
            content_length = len(result.content)
            if current_length + content_length <= max_length:
                limited_results.append(result)
                current_length += content_length
            else:
                break
        
        return limited_results