"""Pinecone vector store provider."""

import logging
import os
from typing import Any, Dict, List, Optional

from agmem.vector_stores.base import VectorStoreBase, VectorSearchResult
from agmem.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class PineconeVectorStore(VectorStoreBase):
    """Pinecone vector store implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        try:
            from pinecone import Pinecone
        except ImportError:
            raise VectorStoreError(
                "pinecone library not installed",
                error_code="VECTOR_PINECONE_001",
                suggestion="Install with: pip install pinecone (not pinecone-client)",
            )
        
        api_key = self.config.get("api_key") or os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise VectorStoreError(
                "Pinecone API key not provided",
                error_code="VECTOR_PINECONE_002",
                suggestion="Set PINECONE_API_KEY env var or pass api_key in config",
            )
        
        index_name = self.config.get("index_name", "agmem")
        self._namespace = self.config.get("namespace")
        embedding_dims = self.config.get("embedding_dims", 768)
        
        pc = Pinecone(api_key=api_key)
        
        # Auto-create index if it doesn't exist
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing_indexes:
            from pinecone import ServerlessSpec
            logger.info(f"Creating Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=embedding_dims,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.config.get("cloud", "aws"),
                    region=self.config.get("region", "us-east-1"),
                ),
            )
        
        self._index = pc.Index(index_name)
    
    async def upsert(
        self,
        id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Insert or update a single vector."""
        try:
            self._index.upsert(
                vectors=[(id, embedding, metadata)],
                namespace=self._namespace,
            )
        except Exception as e:
            logger.error(f"Pinecone upsert failed: {e}")
            raise VectorStoreError(f"Failed to upsert vector: {e}")

    async def upsert_batch(
        self,
        items: List[tuple[str, List[float], Dict[str, Any]]],
    ) -> None:
        """Batch insert/update vectors."""
        if not items:
            return
        
        try:
            vectors = [(id, embedding, metadata) for id, embedding, metadata in items]
            
            # Pinecone recommends batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self._index.upsert(vectors=batch, namespace=self._namespace)
                
        except Exception as e:
            logger.error(f"Pinecone batch upsert failed: {e}")
            raise VectorStoreError(f"Failed to batch upsert vectors: {e}")
    
    async def search(
        self,
        embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        try:
            results = self._index.query(
                vector=embedding,
                top_k=limit,
                filter=filters,
                namespace=self._namespace,
                include_metadata=True,
            )
            
            return [
                VectorSearchResult(
                    id=match.id,
                    score=match.score,
                    metadata=match.metadata or {},
                )
                for match in results.matches
            ]
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            raise VectorStoreError(f"Failed to search vectors: {e}")
    
    async def delete(self, ids: List[str]) -> None:
        """Delete vectors by IDs."""
        if not ids:
            return
        
        try:
            self._index.delete(ids=ids, namespace=self._namespace)
        except Exception as e:
            logger.error(f"Pinecone delete failed: {e}")
            raise VectorStoreError(f"Failed to delete vectors: {e}")
