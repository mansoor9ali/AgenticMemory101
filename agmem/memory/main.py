"""
Main Memory class for AgenticMemory.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from agmem.configs.base import MemoryConfig
from agmem.memory.prompts import FACT_EXTRACTION_PROMPT
from agmem.memory.utils import (
    compute_importance,
    compute_keyword_overlap,
    extract_json,
    is_duplicate,
)
from agmem.models import Memory as MemoryModel
from agmem.models import SearchResult, generate_id
from agmem.utils.factory import EmbedderFactory, LLMFactory, StorageFactory, VectorStoreFactory

logger = logging.getLogger(__name__)


class AsyncMemory:
    """
    Async Memory class for AgenticMemory.

    Provides async methods for memory operations with pluggable backends.
    """

    def __init__(self, config: Optional[Union[MemoryConfig, Dict[str, Any]]] = None):
        """
        Initialize AgenticMemory Memory.

        Args:
            config: MemoryConfig object or dict with configuration
        """
        if config is None:
            config = MemoryConfig()
        elif isinstance(config, dict):
            config = MemoryConfig(**config)

        self.config = config
        self._initialized = False

        # Will be initialized lazily
        self._cache = None
        self._storage = None
        self._vector_store = None
        self._embedder = None
        self._llm = None

    async def _ensure_initialized(self):
        """Ensure all providers are initialized."""
        if self._initialized:
            return

        # Initialize embedder
        self._embedder = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
        )

        # Initialize storage
        self._storage = StorageFactory.create(
            self.config.storage.provider,
            self.config.storage.config,
        )

        # Initialize vector store
        self._vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider,
            self.config.vector_store.config,
        )

        # Initialize LLM
        self._llm = LLMFactory.create(
            self.config.llm.provider,
            self.config.llm.config,
        )

        # Initialize cache if enabled
        if self.config.cache.enabled:
            try:
                from agmem.cache import RedisCache

                self._cache = RedisCache(
                    self.config.cache.redis_url,
                    ttl_seconds=self.config.cache.ttl_seconds,
                    max_per_user=self.config.cache.max_memories_per_user,
                    key_prefix=self.config.cache.key_prefix,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize cache: {e}. Continuing without cache."
                )
                self._cache = None

        self._initialized = True

    async def add(
        self,
        messages: Union[str, List[Dict[str, str]]],
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Add memories from messages or text.

        Args:
            messages: Text string or list of message dicts
            user_id: User identifier
            metadata: Optional metadata to attach
            **kwargs: Additional options (agent_id, run_id, etc.)

        Returns:
            Dict with 'results' containing added memories
        """
        await self._ensure_initialized()

        # Normalize messages to list format
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Extract facts using LLM
        facts = await self._extract_facts(messages)

        if not facts:
            return {"results": []}

        # Get existing memories for deduplication
        existing = await self._storage.query(user_id, limit=100)

        # Process each fact
        results = []
        for fact_data in facts[: self.config.max_facts_per_conversation]:
            # Extract fact text and scoring parameters
            fact_text = fact_data["text"]
            fact_importance = fact_data.get("importance", 0.5)
            fact_decay = fact_data.get("decay", 0.01)

            # Generate embedding
            embedding = self._embedder.embed(fact_text)

            # Check for duplicates
            if is_duplicate(embedding, existing, self.config.dedup_threshold):
                logger.debug(f"Skipping duplicate fact: {fact_text[:50]}...")
                continue

            # Create memory with LLM-assigned importance and decay
            now = datetime.utcnow()
            memory = MemoryModel(
                id=generate_id(),
                user_id=user_id,
                content=fact_text,
                embedding=embedding,
                metadata=metadata or {},
                source="extraction",
                importance=fact_importance,
                decay_rate=fact_decay,
                created_at=now,
                updated_at=now,
                last_accessed_at=now,
            )

            # Save to storage
            await self._storage.save([memory])

            # Save to vector store
            await self._vector_store.upsert(
                id=memory.id,
                embedding=embedding,
                metadata={"user_id": user_id, "content": fact_text},
            )

            # Cache
            if self._cache:
                try:
                    await self._cache.set(memory)
                except Exception as e:
                    logger.warning(f"Cache write failed: {e}")

            results.append(
                {
                    "id": memory.id,
                    "memory": memory.content,
                    "importance": memory.importance,
                    "decay_rate": memory.decay_rate,
                    "event": "ADD",
                }
            )

            # Add to existing for dedup check
            existing.append(memory)

        return {"results": results}

    async def _extract_facts(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Extract facts from messages using LLM.

        Returns list of dicts with keys: text, importance, decay
        """
        # Format messages for prompt
        formatted = "\n".join(
            [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
        )

        response = self._llm.generate_response(
            messages=[
                {"role": "system", "content": FACT_EXTRACTION_PROMPT},
                {"role": "user", "content": formatted},
            ]
        )

        # Parse response
        content = response.get("content", "")
        parsed = extract_json(content)

        if parsed and "facts" in parsed:
            facts = parsed["facts"]
            # Normalize facts to new format
            # (handle both old string format and new dict format)
            normalized = []
            for fact in facts:
                if isinstance(fact, str):
                    # Legacy format - use defaults
                    normalized.append(
                        {
                            "text": fact,
                            "importance": 0.5,
                            "decay": 0.01,
                        }
                    )
                elif isinstance(fact, dict) and "text" in fact:
                    # New format with importance and decay
                    normalized.append(
                        {
                            "text": fact["text"],
                            "importance": float(fact.get("importance", 0.5)),
                            "decay": float(fact.get("decay", 0.01)),
                        }
                    )
            return normalized

        return []

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search for relevant memories.

        Args:
            query: Search query
            user_id: User identifier
            limit: Maximum results to return
            filters: Optional filters

        Returns:
            Dict with 'results' containing matching memories
        """
        await self._ensure_initialized()

        # Generate query embedding
        query_embedding = self._embedder.embed(query)

        # Search vector store
        vector_filters = {"user_id": user_id}
        if filters:
            vector_filters.update(filters)

        vector_results = await self._vector_store.search(
            embedding=query_embedding,
            limit=limit * 3,  # Get more for reranking
            filters=vector_filters,
        )

        if not vector_results:
            return {"results": []}

        # Get full memories from storage
        memory_ids = [r.id for r in vector_results]
        memories = await self._storage.get(memory_ids)

        # Build score map
        score_map = {r.id: r.score for r in vector_results}

        # Score and rank
        now = datetime.now(tz=timezone.utc)
        scored = []
        for memory in memories:
            semantic_score = score_map.get(memory.id, 0.0)
            importance_score = compute_importance(
                memory,
                now,
                self.config.weight_recency,
                self.config.weight_frequency,
                self.config.weight_importance,
            )
            keyword_score = compute_keyword_overlap(query, memory.content)

            final_score = (
                0.5 * semantic_score + 0.3 * importance_score + 0.2 * keyword_score
            )

            scored.append((memory, final_score))

        # Sort and limit
        scored.sort(key=lambda x: x[1], reverse=True)
        top_memories = scored[:limit]

        # Update access patterns synchronously to avoid race conditions on close
        try:
            await self._update_access([m for m, _ in top_memories])
        except Exception as e:
            logger.warning(f"Failed to update access patterns: {e}")

        # Format results
        results = []
        for memory, score in top_memories:
            results.append(
                SearchResult(
                    id=memory.id,
                    memory=memory.content,
                    score=score,
                    metadata=memory.metadata,
                    created_at=(
                        memory.created_at.isoformat() if memory.created_at else None
                    ),
                    updated_at=(
                        memory.updated_at.isoformat() if memory.updated_at else None
                    ),
                ).to_dict()
            )

        return {"results": results}

    async def _update_access(self, memories: List[MemoryModel]) -> None:
        """Update access count and timestamp for retrieved memories."""
        now = datetime.utcnow()
        for memory in memories:
            memory.last_accessed_at = now
            memory.access_count += 1
            await self._storage.update(memory)

        # Promote to cache
        if self._cache:
            try:
                await self._cache.promote(memories)
            except Exception as e:
                logger.warning(f"Cache promote failed: {e}")

    async def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a single memory by ID."""
        await self._ensure_initialized()

        # Try cache first
        if self._cache:
            try:
                memory = await self._cache.get(memory_id)
                if memory:
                    return memory.to_dict()
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")

        # Fall back to storage
        memories = await self._storage.get([memory_id])
        if memories:
            return memories[0].to_dict()

        return None

    async def get_all(
        self,
        user_id: str,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get all memories for a user."""
        await self._ensure_initialized()

        memories = await self._storage.query(user_id, limit=limit)

        return {
            "results": [
                {
                    "id": m.id,
                    "memory": m.content,
                    "metadata": m.metadata,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "updated_at": m.updated_at.isoformat() if m.updated_at else None,
                }
                for m in memories
            ]
        }

    async def update(
        self,
        memory_id: str,
        content: str,
    ) -> Dict[str, Any]:
        """Update a memory's content."""
        await self._ensure_initialized()

        memories = await self._storage.get([memory_id])
        if not memories:
            return {"success": False, "message": "Memory not found"}

        memory = memories[0]
        memory.content = content
        memory.updated_at = datetime.utcnow()

        # Re-embed
        memory.embedding = self._embedder.embed(content)

        # Update storage
        await self._storage.update(memory)

        # Update vector store
        await self._vector_store.upsert(
            id=memory.id,
            embedding=memory.embedding,
            metadata={"user_id": memory.user_id, "content": content},
        )

        # Update cache
        if self._cache:
            try:
                await self._cache.set(memory)
            except Exception as e:
                logger.warning(f"Cache update failed: {e}")

        return {"success": True, "message": "Memory updated"}

    async def delete(self, memory_id: str) -> Dict[str, Any]:
        """Delete a memory."""
        await self._ensure_initialized()

        # Get memory to find user_id
        memories = await self._storage.get([memory_id])
        if not memories:
            return {"success": False, "message": "Memory not found"}

        user_id = memories[0].user_id

        # Delete from all stores
        await self._storage.delete([memory_id])
        await self._vector_store.delete([memory_id])

        if self._cache:
            try:
                await self._cache.delete(memory_id, user_id)
            except Exception as e:
                logger.warning(f"Cache delete failed: {e}")

        return {"success": True, "message": "Memory deleted"}

    async def delete_all(self, user_id: str) -> Dict[str, Any]:
        """Delete all memories for a user."""
        await self._ensure_initialized()

        # Get all memory IDs
        memories = await self._storage.query(user_id, limit=10000)
        memory_ids = [m.id for m in memories]

        if memory_ids:
            await self._storage.delete(memory_ids)
            await self._vector_store.delete(memory_ids)

        if self._cache:
            try:
                await self._cache.clear_user(user_id)
            except Exception as e:
                logger.warning(f"Cache clear failed: {e}")

        return {"success": True, "message": f"Deleted {len(memory_ids)} memories"}

    async def close(self) -> None:
        """Close all connections."""
        if self._cache:
            await self._cache.close()
        if self._storage:
            await self._storage.close()
        if self._vector_store:
            await self._vector_store.close()


class Memory:
    """
    Synchronous wrapper for AsyncMemory.

    Provides sync methods that wrap the async implementation.
    """

    def __init__(self, config: Optional[Union[MemoryConfig, Dict[str, Any]]] = None):
        self._async_memory = AsyncMemory(config)
        self._loop = None

    def _get_loop(self):
        """Get or create event loop."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
            return self._loop

    def _run(self, coro):
        """Run coroutine synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(coro)

    def add(
        self,
        messages: Union[str, List[Dict[str, str]]],
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add memories from messages or text."""
        return self._run(self._async_memory.add(messages, user_id, metadata, **kwargs))

    def search(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Search for relevant memories."""
        return self._run(self._async_memory.search(query, user_id, limit, filters))

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a single memory by ID."""
        return self._run(self._async_memory.get(memory_id))

    def get_all(self, user_id: str, limit: int = 100) -> Dict[str, Any]:
        """Get all memories for a user."""
        return self._run(self._async_memory.get_all(user_id, limit))

    def update(self, memory_id: str, content: str) -> Dict[str, Any]:
        """Update a memory's content."""
        return self._run(self._async_memory.update(memory_id, content))

    def delete(self, memory_id: str) -> Dict[str, Any]:
        """Delete a memory."""
        return self._run(self._async_memory.delete(memory_id))

    def delete_all(self, user_id: str) -> Dict[str, Any]:
        """Delete all memories for a user."""
        return self._run(self._async_memory.delete_all(user_id))

    def close(self) -> None:
        """Close all connections."""
        self._run(self._async_memory.close())
        if self._loop:
            self._loop.close()
