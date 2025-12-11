"""Chroma vector store provider for local/dev use."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agmem.vector_stores.base import VectorStoreBase, VectorSearchResult
from agmem.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStoreBase):
    """Chroma vector store implementation for local development."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        try:
            import chromadb
        except ImportError:
            raise VectorStoreError(
                "chromadb library not installed",
                error_code="VECTOR_CHROMA_001",
                suggestion="Install with: pip install chromadb",
            )
        
        self._path = os.path.expanduser(
            self.config.get("path", "~/.agmem/chroma")
        )
        self._collection_name = self.config.get("collection_name", "agmem")
        
        # Create directory if needed
        Path(self._path).mkdir(parents=True, exist_ok=True)
        
        # Initialize client
        self._client = chromadb.PersistentClient(path=self._path)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def upsert(
        self,
        id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Insert or update a single vector."""
        try:
            self._collection.upsert(
                ids=[id],
                embeddings=[embedding],
                metadatas=[metadata],
            )
        except Exception as e:
            logger.error(f"Chroma upsert failed: {e}")
            raise VectorStoreError(
                f"Failed to upsert vector: {e}",
                error_code="VECTOR_CHROMA_002",
            )
    
    async def upsert_batch(
        self,
        items: List[tuple[str, List[float], Dict[str, Any]]],
    ) -> None:
        """Batch insert/update vectors."""
        if not items:
            return
        
        try:
            ids = [item[0] for item in items]
            embeddings = [item[1] for item in items]
            metadatas = [item[2] for item in items]
            
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as e:
            logger.error(f"Chroma batch upsert failed: {e}")
            raise VectorStoreError(
                f"Failed to batch upsert vectors: {e}",
                error_code="VECTOR_CHROMA_003",
            )

    async def search(
        self,
        embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        try:
            # Build where clause for filters
            where = None
            if filters:
                where = {}
                for key, value in filters.items():
                    where[key] = value
            
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=limit,
                where=where,
            )
            
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i, id in enumerate(results["ids"][0]):
                    score = 1.0 - results["distances"][0][i] if results["distances"] else 0.0
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    search_results.append(VectorSearchResult(
                        id=id,
                        score=score,
                        metadata=metadata,
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Chroma search failed: {e}")
            raise VectorStoreError(
                f"Failed to search vectors: {e}",
                error_code="VECTOR_CHROMA_004",
            )
    
    async def delete(self, ids: List[str]) -> None:
        """Delete vectors by IDs."""
        if not ids:
            return
        
        try:
            self._collection.delete(ids=ids)
        except Exception as e:
            logger.error(f"Chroma delete failed: {e}")
            raise VectorStoreError(
                f"Failed to delete vectors: {e}",
                error_code="VECTOR_CHROMA_005",
            )
