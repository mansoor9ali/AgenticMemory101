"""SQLite storage provider for local/dev use."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agmem.storage.base import StorageBase
from agmem.models import Memory
from agmem.exceptions import StorageError

logger = logging.getLogger(__name__)


class SQLiteStorage(StorageBase):
    """SQLite storage implementation for local development."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        try:
            import aiosqlite
        except ImportError:
            raise StorageError(
                "aiosqlite library not installed",
                error_code="STORAGE_SQLITE_001",
                suggestion="Install with: pip install aiosqlite",
            )
        
        self._path = os.path.expanduser(
            self.config.get("path", "~/.agmem/memories.db")
        )
        self._collection = self.config.get("collection_name", "memories")
        self._db = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure database and table exist."""
        if self._initialized:
            return
        
        import aiosqlite
        
        # Create directory if needed
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        
        self._db = await aiosqlite.connect(self._path)
        
        # Create table (embeddings stored in vector store, not here)
        await self._db.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._collection} (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                hash TEXT,
                metadata TEXT,
                tags TEXT,
                source TEXT,
                importance REAL,
                decay_rate REAL,
                access_count INTEGER,
                created_at TEXT,
                updated_at TEXT,
                last_accessed_at TEXT,
                links TEXT,
                is_consolidated INTEGER,
                consolidated_from TEXT
            )
        """)
        
        # Create index on user_id
        await self._db.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._collection}_user_id 
            ON {self._collection}(user_id)
        """)
        
        await self._db.commit()
        self._initialized = True

    async def save(self, memories: List[Memory]) -> None:
        """Save memories to SQLite (embeddings stored in vector store)."""
        await self._ensure_initialized()
        
        for memory in memories:
            await self._db.execute(f"""
                INSERT OR REPLACE INTO {self._collection}
                (id, user_id, content, hash, metadata, tags, source,
                 importance, decay_rate, access_count, created_at, updated_at,
                 last_accessed_at, links, is_consolidated, consolidated_from)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id,
                memory.user_id,
                memory.content,
                memory.hash,
                json.dumps(memory.metadata),
                json.dumps(memory.tags),
                memory.source,
                memory.importance,
                memory.decay_rate,
                memory.access_count,
                memory.created_at.isoformat() if memory.created_at else None,
                memory.updated_at.isoformat() if memory.updated_at else None,
                memory.last_accessed_at.isoformat() if memory.last_accessed_at else None,
                json.dumps(memory.links),
                1 if memory.is_consolidated else 0,
                json.dumps(memory.consolidated_from),
            ))
        
        await self._db.commit()
    
    async def get(self, ids: List[str]) -> List[Memory]:
        """Get memories by IDs."""
        await self._ensure_initialized()
        
        if not ids:
            return []
        
        placeholders = ",".join("?" * len(ids))
        cursor = await self._db.execute(
            f"SELECT * FROM {self._collection} WHERE id IN ({placeholders})",
            ids
        )
        rows = await cursor.fetchall()
        
        return [self._row_to_memory(row) for row in rows]

    def _row_to_memory(self, row) -> Memory:
        """Convert database row to Memory object (embeddings in vector store)."""
        return Memory(
            id=row[0],
            user_id=row[1],
            content=row[2],
            hash=row[3] or "",
            embedding=None,  # Embeddings stored in vector store, not here
            metadata=json.loads(row[4]) if row[4] else {},
            tags=json.loads(row[5]) if row[5] else [],
            source=row[6] or "manual",
            importance=row[7] or 0.5,
            decay_rate=row[8] or 0.01,
            access_count=row[9] or 0,
            created_at=datetime.fromisoformat(row[10]) if row[10] else datetime.utcnow(),
            updated_at=datetime.fromisoformat(row[11]) if row[11] else datetime.utcnow(),
            last_accessed_at=datetime.fromisoformat(row[12]) if row[12] else datetime.utcnow(),
            links=json.loads(row[13]) if row[13] else [],
            is_consolidated=bool(row[14]),
            consolidated_from=json.loads(row[15]) if row[15] else [],
        )
    
    async def delete(self, ids: List[str]) -> None:
        """Delete memories by IDs."""
        await self._ensure_initialized()
        
        if not ids:
            return
        
        placeholders = ",".join("?" * len(ids))
        await self._db.execute(
            f"DELETE FROM {self._collection} WHERE id IN ({placeholders})",
            ids
        )
        await self._db.commit()
    
    async def query(
        self,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Memory]:
        """Query memories with optional filters."""
        await self._ensure_initialized()
        
        cursor = await self._db.execute(
            f"SELECT * FROM {self._collection} WHERE user_id = ? LIMIT ?",
            (user_id, limit)
        )
        rows = await cursor.fetchall()
        
        return [self._row_to_memory(row) for row in rows]

    async def update(self, memory: Memory) -> None:
        """Update a single memory."""
        await self.save([memory])
    
    async def count(self, user_id: str) -> int:
        """Count memories for a user."""
        await self._ensure_initialized()
        
        cursor = await self._db.execute(
            f"SELECT COUNT(*) FROM {self._collection} WHERE user_id = ?",
            (user_id,)
        )
        row = await cursor.fetchone()
        return row[0] if row else 0
    
    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
