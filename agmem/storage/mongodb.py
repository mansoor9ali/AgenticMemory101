"""MongoDB storage provider."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from agmem.storage.base import StorageBase
from agmem.models import Memory
from agmem.exceptions import StorageError

logger = logging.getLogger(__name__)


class MongoDBStorage(StorageBase):
    """MongoDB storage implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError:
            raise StorageError(
                "motor library not installed",
                error_code="STORAGE_MONGO_001",
                suggestion="Install with: pip install motor",
            )
        
        uri = self.config.get("uri", "mongodb://localhost:27017")
        database = self.config.get("database", "agmem")
        collection = self.config.get("collection_name", "memories")
        
        self._client = AsyncIOMotorClient(uri)
        self._db = self._client[database]
        self._collection = self._db[collection]
    
    async def save(self, memories: List[Memory]) -> None:
        """Save memories to MongoDB."""
        from pymongo import UpdateOne
        
        operations = []
        for memory in memories:
            operations.append(
                UpdateOne(
                    {"_id": memory.id},
                    {"$set": self._memory_to_doc(memory)},
                    upsert=True
                )
            )
        
        if operations:
            await self._collection.bulk_write(operations)
    
    async def get(self, ids: List[str]) -> List[Memory]:
        """Get memories by IDs."""
        if not ids:
            return []
        
        cursor = self._collection.find({"_id": {"$in": ids}})
        docs = await cursor.to_list(length=len(ids))
        
        return [self._doc_to_memory(doc) for doc in docs]
    
    async def delete(self, ids: List[str]) -> None:
        """Delete memories by IDs."""
        if not ids:
            return
        
        await self._collection.delete_many({"_id": {"$in": ids}})

    async def query(
        self,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Memory]:
        """Query memories with optional filters."""
        query = {"user_id": user_id}
        if filters:
            query.update(filters)
        
        cursor = self._collection.find(query).limit(limit)
        docs = await cursor.to_list(length=limit)
        
        return [self._doc_to_memory(doc) for doc in docs]
    
    async def update(self, memory: Memory) -> None:
        """Update a single memory."""
        await self.save([memory])
    
    async def count(self, user_id: str) -> int:
        """Count memories for a user."""
        return await self._collection.count_documents({"user_id": user_id})
    
    async def close(self) -> None:
        """Close MongoDB connection."""
        self._client.close()
    
    def _memory_to_doc(self, memory: Memory) -> Dict[str, Any]:
        """Convert Memory to MongoDB document (embeddings stored in vector store)."""
        return {
            "_id": memory.id,
            "user_id": memory.user_id,
            "content": memory.content,
            "hash": memory.hash,
            # embedding not stored here - stored in vector store
            "metadata": memory.metadata,
            "tags": memory.tags,
            "source": memory.source,
            "importance": memory.importance,
            "decay_rate": memory.decay_rate,
            "access_count": memory.access_count,
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
            "last_accessed_at": memory.last_accessed_at,
            "links": memory.links,
            "is_consolidated": memory.is_consolidated,
            "consolidated_from": memory.consolidated_from,
        }
    
    def _doc_to_memory(self, doc: Dict[str, Any]) -> Memory:
        """Convert MongoDB document to Memory (embeddings in vector store)."""
        return Memory(
            id=doc["_id"],
            user_id=doc["user_id"],
            content=doc["content"],
            hash=doc.get("hash", ""),
            embedding=None,  # Embeddings stored in vector store, not here
            metadata=doc.get("metadata", {}),
            tags=doc.get("tags", []),
            source=doc.get("source", "manual"),
            importance=doc.get("importance", 0.5),
            decay_rate=doc.get("decay_rate", 0.01),
            access_count=doc.get("access_count", 0),
            created_at=doc.get("created_at", datetime.utcnow()),
            updated_at=doc.get("updated_at", datetime.utcnow()),
            last_accessed_at=doc.get("last_accessed_at", datetime.utcnow()),
            links=doc.get("links", []),
            is_consolidated=doc.get("is_consolidated", False),
            consolidated_from=doc.get("consolidated_from", []),
        )
