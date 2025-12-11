"""Qdrant vector store provider."""

import logging
from typing import Any, Dict, List, Optional

from agmem.vector_stores.base import VectorStoreBase, VectorSearchResult
from agmem.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStoreBase):
    """Qdrant vector store implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise VectorStoreError(
                "qdrant-client library not installed",
                error_code="VECTOR_QDRANT_001",
                suggestion="Install with: pip install qdrant-client",
            )
        
        url = self.config.get("url", "http://localhost:6333")
        api_key = self.config.get("api_key")
        self._collection = self.config.get("collection_name", "agmem")
        self._dims = self.config.get("embedding_dims", 1536)
        
        self._client = QdrantClient(url=url, api_key=api_key)
        
        # Create collection if not exists
        try:
            self._client.get_collection(self._collection)
        except Exception:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dims,
                    distance=Distance.COSINE,
                ),
            )
    
    async def upsert(
        self,
        id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Insert or update a single vector."""
        from qdrant_client.models import PointStruct
        
        try:
            self._client.upsert(
                collection_name=self._collection,
                points=[
                    PointStruct(
                        id=id,
                        vector=embedding,
                        payload=metadata,
                    )
                ],
            )
        except Exception as e:
            logger.error(f"Qdrant upsert failed: {e}")
            raise VectorStoreError(f"Failed to upsert vector: {e}")

    async def upsert_batch(
        self,
        items: List[tuple[str, List[float], Dict[str, Any]]],
    ) -> None:
        """Batch insert/update vectors."""
        from qdrant_client.models import PointStruct
        
        if not items:
            return
        
        try:
            points = [
                PointStruct(id=id, vector=embedding, payload=metadata)
                for id, embedding, metadata in items
            ]
            self._client.upsert(
                collection_name=self._collection,
                points=points,
            )
        except Exception as e:
            logger.error(f"Qdrant batch upsert failed: {e}")
            raise VectorStoreError(f"Failed to batch upsert vectors: {e}")
    
    async def search(
        self,
        embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        try:
            # Build filter
            qdrant_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                qdrant_filter = Filter(must=conditions)
            
            results = self._client.query_points(
                collection_name=self._collection,
                query=embedding,
                limit=limit,
                query_filter=qdrant_filter,
            ).points
            
            return [
                VectorSearchResult(
                    id=str(r.id),
                    score=r.score,
                    metadata=r.payload or {},
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            raise VectorStoreError(f"Failed to search vectors: {e}")
    
    async def delete(self, ids: List[str]) -> None:
        """Delete vectors by IDs."""
        if not ids:
            return
        
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=ids,
            )
        except Exception as e:
            logger.error(f"Qdrant delete failed: {e}")
            raise VectorStoreError(f"Failed to delete vectors: {e}")
