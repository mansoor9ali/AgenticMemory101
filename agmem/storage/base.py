"""Base class for L2 storage providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agmem.models import Memory


class StorageBase(ABC):
    """Abstract base class for L2 storage providers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    async def save(self, memories: List[Memory]) -> None:
        """Save memories to storage."""
        pass
    
    @abstractmethod
    async def get(self, ids: List[str]) -> List[Memory]:
        """Get memories by IDs."""
        pass
    
    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """Delete memories by IDs."""
        pass
    
    @abstractmethod
    async def query(
        self,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Memory]:
        """Query memories with optional filters."""
        pass
    
    @abstractmethod
    async def update(self, memory: Memory) -> None:
        """Update a single memory."""
        pass
    
    @abstractmethod
    async def count(self, user_id: str) -> int:
        """Count memories for a user."""
        pass
    
    async def close(self) -> None:
        """Close connections. Override if needed."""
        pass
