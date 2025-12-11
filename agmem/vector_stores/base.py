"""Base class for vector store providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    id: str
    score: float
    metadata: Dict[str, Any]


class VectorStoreBase(ABC):
    """Abstract base class for vector store providers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    async def upsert(
        self,
        id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Insert or update a single vector."""
        pass
    
    @abstractmethod
    async def upsert_batch(
        self,
        items: List[tuple[str, List[float], Dict[str, Any]]],
    ) -> None:
        """Batch insert/update vectors. Items are (id, embedding, metadata) tuples."""
        pass
    
    @abstractmethod
    async def search(
        self,
        embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """Delete vectors by IDs."""
        pass
    
    async def close(self) -> None:
        """Close connections. Override if needed."""
        pass
