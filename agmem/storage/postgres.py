"""PostgreSQL storage provider."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from agmem.storage.base import StorageBase
from agmem.models import Memory
from agmem.exceptions import StorageError

logger = logging.getLogger(__name__)


class PostgresStorage(StorageBase):
    """PostgreSQL storage implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        try:
            import asyncpg
        except ImportError:
            raise StorageError(
                "asyncpg library not installed",
                error_code="STORAGE_PG_001",
                suggestion="Install with: pip install asyncpg",
            )
        
        self._pool = None
        self._collection = self.config.get("collection_name", "memories")
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure connection pool and table exist."""
        if self._initialized:
            return
        
        import asyncpg
        
        self._pool = await asyncpg.create_pool(
            host=self.config.get("host", "localhost"),
            port=self.config.get("port", 5432),
            database=self.config.get("database", "agmem"),
            user=self.config["user"],
            password=self.config["password"],
            min_size=self.config.get("min_connections", 1),
            max_size=self.config.get("max_connections", 10),
        )
        
        # Create table (embeddings stored in vector store, not here)
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._collection} (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    hash TEXT,
                    metadata JSONB DEFAULT '{{}}',
                    tags JSONB DEFAULT '[]',
                    source TEXT DEFAULT 'manual',
                    importance REAL DEFAULT 0.5,
                    decay_rate REAL DEFAULT 0.01,
                    access_count INTEGER DEFAULT 0,
                    created_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ,
                    last_accessed_at TIMESTAMPTZ,
                    links JSONB DEFAULT '[]',
                    is_consolidated BOOLEAN DEFAULT FALSE,
                    consolidated_from JSONB DEFAULT '[]'
                )
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._collection}_user_id 
                ON {self._collection}(user_id)
            """)
        
        self._initialized = True

    async def save(self, memories: List[Memory]) -> None:
        """Save memories to PostgreSQL (embeddings stored in vector store)."""
        await self._ensure_initialized()
        
        async with self._pool.acquire() as conn:
            for memory in memories:
                await conn.execute(f"""
                    INSERT INTO {self._collection}
                    (id, user_id, content, hash, metadata, tags, source,
                     importance, decay_rate, access_count, created_at, updated_at,
                     last_accessed_at, links, is_consolidated, consolidated_from)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at,
                        last_accessed_at = EXCLUDED.last_accessed_at,
                        access_count = EXCLUDED.access_count
                """,
                    memory.id, memory.user_id, memory.content, memory.hash,
                    json.dumps(memory.metadata), json.dumps(memory.tags), memory.source,
                    memory.importance, memory.decay_rate, memory.access_count,
                    memory.created_at, memory.updated_at, memory.last_accessed_at,
                    json.dumps(memory.links), memory.is_consolidated,
                    json.dumps(memory.consolidated_from),
                )
    
    async def get(self, ids: List[str]) -> List[Memory]:
        """Get memories by IDs."""
        await self._ensure_initialized()
        
        if not ids:
            return []
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {self._collection} WHERE id = ANY($1)",
                ids
            )
        
        return [self._row_to_memory(row) for row in rows]
    
    async def delete(self, ids: List[str]) -> None:
        """Delete memories by IDs."""
        await self._ensure_initialized()
        
        if not ids:
            return
        
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._collection} WHERE id = ANY($1)",
                ids
            )

    async def query(
        self,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Memory]:
        """Query memories with optional filters."""
        await self._ensure_initialized()
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {self._collection} WHERE user_id = $1 LIMIT $2",
                user_id, limit
            )
        
        return [self._row_to_memory(row) for row in rows]
    
    async def update(self, memory: Memory) -> None:
        """Update a single memory."""
        await self.save([memory])
    
    async def count(self, user_id: str) -> int:
        """Count memories for a user."""
        await self._ensure_initialized()
        
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                f"SELECT COUNT(*) FROM {self._collection} WHERE user_id = $1",
                user_id
            )
        return result or 0
    
    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
    
    def _row_to_memory(self, row) -> Memory:
        """Convert database row to Memory object (embeddings fetched from vector store)."""
        return Memory(
            id=row["id"],
            user_id=row["user_id"],
            content=row["content"],
            hash=row["hash"] or "",
            embedding=None,  # Embeddings stored in vector store, not here
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            tags=json.loads(row["tags"]) if row["tags"] else [],
            source=row["source"] or "manual",
            importance=row["importance"] or 0.5,
            decay_rate=row["decay_rate"] or 0.01,
            access_count=row["access_count"] or 0,
            created_at=row["created_at"] or datetime.utcnow(),
            updated_at=row["updated_at"] or datetime.utcnow(),
            last_accessed_at=row["last_accessed_at"] or datetime.utcnow(),
            links=json.loads(row["links"]) if row["links"] else [],
            is_consolidated=row["is_consolidated"] or False,
            consolidated_from=json.loads(row["consolidated_from"]) if row["consolidated_from"] else [],
        )
