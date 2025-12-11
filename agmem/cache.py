"""
Redis L1 cache implementation - the fixed caching layer.
"""

import json
import logging
from datetime import datetime
from typing import List, Optional

from agmem.models import Memory
from agmem.exceptions import CacheError

logger = logging.getLogger(__name__)


class RedisCache:
    """L1 cache using Redis for hot/recent memories."""
    
    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 3600,
        max_per_user: int = 100,
        key_prefix: str = "agmem",
    ):
        try:
            import redis.asyncio as redis
        except ImportError:
            raise CacheError(
                "redis library not installed",
                error_code="CACHE_REDIS_001",
                suggestion="Install with: pip install redis",
            )
        
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl_seconds
        self.max_per_user = max_per_user
        self.prefix = key_prefix

    def _user_key(self, user_id: str) -> str:
        return f"{self.prefix}:memories:{user_id}"
    
    def _memory_key(self, memory_id: str) -> str:
        return f"{self.prefix}:memory:{memory_id}"
    
    async def get(self, memory_id: str) -> Optional[Memory]:
        """Get a single memory from cache."""
        try:
            data = await self.client.get(self._memory_key(memory_id))
            if data:
                return Memory.from_dict(json.loads(data))
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        return None
    
    async def get_many(self, memory_ids: List[str]) -> List[Memory]:
        """Get multiple memories from cache."""
        if not memory_ids:
            return []
        
        try:
            keys = [self._memory_key(mid) for mid in memory_ids]
            values = await self.client.mget(keys)
            
            memories = []
            for val in values:
                if val:
                    memories.append(Memory.from_dict(json.loads(val)))
            return memories
        except Exception as e:
            logger.warning(f"Cache get_many failed: {e}")
            return []

    async def get_recent(self, user_id: str, limit: int = 10) -> List[Memory]:
        """Get most recently accessed memories for a user."""
        try:
            key = self._user_key(user_id)
            memory_ids = await self.client.zrevrange(key, 0, limit - 1)
            
            if not memory_ids:
                return []
            
            return await self.get_many(memory_ids)
        except Exception as e:
            logger.warning(f"Cache get_recent failed: {e}")
            return []
    
    async def set(self, memory: Memory) -> None:
        """Cache a memory."""
        try:
            memory_key = self._memory_key(memory.id)
            user_key = self._user_key(memory.user_id)
            
            # Store the memory data
            await self.client.set(
                memory_key,
                json.dumps(memory.to_dict()),
                ex=self.ttl
            )
            
            # Add to user's sorted set (score = timestamp)
            score = memory.last_accessed_at.timestamp()
            await self.client.zadd(user_key, {memory.id: score})
            
            # Trim to max size
            await self.client.zremrangebyrank(user_key, 0, -(self.max_per_user + 1))
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    async def set_many(self, memories: List[Memory]) -> None:
        """Cache multiple memories."""
        if not memories:
            return
        
        try:
            pipe = self.client.pipeline()
            
            for memory in memories:
                memory_key = self._memory_key(memory.id)
                user_key = self._user_key(memory.user_id)
                
                pipe.set(memory_key, json.dumps(memory.to_dict()), ex=self.ttl)
                pipe.zadd(user_key, {memory.id: memory.last_accessed_at.timestamp()})
            
            await pipe.execute()
        except Exception as e:
            logger.warning(f"Cache set_many failed: {e}")
    
    async def promote(self, memories: List[Memory]) -> None:
        """Promote memories to L1 (update access time and cache)."""
        now = datetime.utcnow()
        for memory in memories:
            memory.last_accessed_at = now
            memory.access_count += 1
        
        await self.set_many(memories)
    
    async def delete(self, memory_id: str, user_id: str) -> None:
        """Remove a memory from cache."""
        try:
            await self.client.delete(self._memory_key(memory_id))
            await self.client.zrem(self._user_key(user_id), memory_id)
        except Exception as e:
            logger.warning(f"Cache delete failed: {e}")

    async def delete_many(self, memory_ids: List[str], user_id: str) -> None:
        """Remove multiple memories from cache."""
        if not memory_ids:
            return
        
        try:
            pipe = self.client.pipeline()
            for mid in memory_ids:
                pipe.delete(self._memory_key(mid))
            pipe.zrem(self._user_key(user_id), *memory_ids)
            await pipe.execute()
        except Exception as e:
            logger.warning(f"Cache delete_many failed: {e}")
    
    async def clear_user(self, user_id: str) -> None:
        """Clear all cached memories for a user."""
        try:
            key = self._user_key(user_id)
            memory_ids = await self.client.zrange(key, 0, -1)
            
            if memory_ids:
                pipe = self.client.pipeline()
                for mid in memory_ids:
                    pipe.delete(self._memory_key(mid))
                pipe.delete(key)
                await pipe.execute()
        except Exception as e:
            logger.warning(f"Cache clear_user failed: {e}")
    
    async def close(self) -> None:
        """Close the Redis connection."""
        try:
            await self.client.close()
        except Exception as e:
            logger.warning(f"Cache close failed: {e}")
