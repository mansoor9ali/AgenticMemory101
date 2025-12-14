"""
Multi-Tenant Memory Module for AgenticMemory.

Provides memory isolation by userId, agentId, or sessionId.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
import logging
import os

from agmem import AsyncMemory, AsyncGraphMemory

logger = logging.getLogger(__name__)


class TenantType(Enum):
    """Types of tenant isolation."""
    USER = "user"       # Persistent user memory
    AGENT = "agent"     # Shared across users (agent knowledge)
    SESSION = "session" # Temporary session memory
    COMBINED = "combined"  # User + Agent combination


class TenantId:
    """
    Represents a tenant identifier for memory isolation.

    Examples:
        - user:alice -> User-specific persistent memory
        - agent:support-bot -> Agent-shared memory
        - session:sess-123 -> Temporary session memory
        - user:alice:agent:support-bot -> User+Agent specific memory
    """

    def __init__(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.user_id = user_id
        self.agent_id = agent_id
        self.session_id = session_id
        self._tenant_id, self._tenant_type = self._build()

    def _build(self) -> tuple[str, TenantType]:
        """Build the tenant identifier string."""
        if self.session_id:
            # Session memory (temporary, highest isolation)
            return f"session:{self.session_id}", TenantType.SESSION
        elif self.user_id and self.agent_id:
            # User + Agent combination (user-specific agent context)
            return f"user:{self.user_id}:agent:{self.agent_id}", TenantType.COMBINED
        elif self.agent_id:
            # Agent memory (shared across users)
            return f"agent:{self.agent_id}", TenantType.AGENT
        elif self.user_id:
            # User memory (persistent)
            return f"user:{self.user_id}", TenantType.USER
        else:
            raise ValueError(
                "At least one of user_id, agent_id, or session_id is required"
            )

    @property
    def id(self) -> str:
        """Get the tenant ID string."""
        return self._tenant_id

    @property
    def type(self) -> TenantType:
        """Get the tenant type."""
        return self._tenant_type

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for metadata."""
        return {
            "tenant_id": self._tenant_id,
            "tenant_type": self._tenant_type.value,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
        }

    def __str__(self) -> str:
        return self._tenant_id

    def __repr__(self) -> str:
        return f"TenantId({self._tenant_id}, type={self._tenant_type.value})"


class MultiTenantMemory:
    """
    Multi-Tenant Memory wrapper with isolation by userId, agentId, or sessionId.

    Usage Examples:
        # User memory (persistent)
        await memory.add("I prefer dark mode", user_id="alice")

        # Agent memory (shared across users)
        await memory.add("Company policy: respond within 24 hours", agent_id="support-bot")

        # Session memory (temporary)
        await memory.add("Currently discussing order #12345", session_id="sess-123")

        # User + Agent combination
        await memory.add("Alice prefers email responses", user_id="alice", agent_id="support-bot")
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multi-tenant memory.

        Args:
            config: Configuration dictionary for AsyncMemory
        """
        self.config = config
        self._memory: Optional[AsyncMemory] = None
        self._enable_isolation = os.getenv("ENABLE_TENANT_ISOLATION", "true").lower() == "true"

    async def _get_memory(self) -> AsyncMemory:
        """Get or create the memory instance."""
        if self._memory is None:
            self._memory = AsyncMemory(self.config)
        return self._memory

    def _build_tenant(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> TenantId:
        """Build a TenantId from the provided identifiers."""
        return TenantId(user_id=user_id, agent_id=agent_id, session_id=session_id)

    async def add(
        self,
        messages: Union[str, List[Dict[str, str]]],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Add memories with tenant isolation.

        Args:
            messages: Text string or list of message dicts
            user_id: User identifier (persistent memory)
            agent_id: Agent identifier (shared memory)
            session_id: Session identifier (temporary memory)
            metadata: Optional metadata to attach
            **kwargs: Additional options

        Returns:
            Dict with 'results' containing added memories
        """
        tenant = self._build_tenant(user_id, agent_id, session_id)

        # Enrich metadata with tenant info
        enriched_metadata = metadata.copy() if metadata else {}
        enriched_metadata.update(tenant.to_dict())

        memory = await self._get_memory()
        result = await memory.add(
            messages=messages,
            user_id=tenant.id,
            metadata=enriched_metadata,
            **kwargs,
        )

        # Add tenant info to result
        result["tenant"] = tenant.to_dict()
        return result

    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search memories within tenant scope.

        Args:
            query: Search query
            user_id: User identifier
            agent_id: Agent identifier
            session_id: Session identifier
            limit: Maximum results to return
            filters: Optional additional filters

        Returns:
            Dict with 'results' containing matching memories
        """
        tenant = self._build_tenant(user_id, agent_id, session_id)

        memory = await self._get_memory()
        result = await memory.search(
            query=query,
            user_id=tenant.id,
            limit=limit,
            filters=filters,
        )

        result["tenant"] = tenant.to_dict()
        return result

    async def get_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Get all memories for a tenant.

        Args:
            user_id: User identifier
            agent_id: Agent identifier
            session_id: Session identifier
            limit: Maximum results

        Returns:
            Dict with 'results' containing all memories
        """
        tenant = self._build_tenant(user_id, agent_id, session_id)

        memory = await self._get_memory()
        result = await memory.get_all(user_id=tenant.id, limit=limit)

        result["tenant"] = tenant.to_dict()
        return result

    async def delete_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Delete all memories for a tenant.

        Args:
            user_id: User identifier
            agent_id: Agent identifier
            session_id: Session identifier

        Returns:
            Dict with success status
        """
        tenant = self._build_tenant(user_id, agent_id, session_id)

        memory = await self._get_memory()
        result = await memory.delete_all(user_id=tenant.id)

        result["tenant"] = tenant.to_dict()
        return result

    async def close(self) -> None:
        """Close all connections."""
        if self._memory:
            await self._memory.close()
            self._memory = None


class MultiTenantGraphMemory:
    """
    Multi-Tenant Graph Memory wrapper with isolation by userId, agentId, or sessionId.

    Similar to MultiTenantMemory but for graph-based memory.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multi-tenant graph memory.

        Args:
            config: Configuration dictionary for AsyncGraphMemory
        """
        self.config = config
        self._graph_memory: Optional[AsyncGraphMemory] = None

    async def _get_memory(self) -> AsyncGraphMemory:
        """Get or create the graph memory instance."""
        if self._graph_memory is None:
            self._graph_memory = AsyncGraphMemory(self.config)
        return self._graph_memory

    def _build_tenant(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> TenantId:
        """Build a TenantId from the provided identifiers."""
        return TenantId(user_id=user_id, agent_id=agent_id, session_id=session_id)

    async def add(
        self,
        messages: Union[str, List[Dict[str, str]]],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships with tenant isolation.

        Args:
            messages: Text string or list of message dicts
            user_id: User identifier
            agent_id: Agent identifier
            session_id: Session identifier
            metadata: Optional metadata

        Returns:
            Dict with 'entities' and 'relationships' added
        """
        tenant = self._build_tenant(user_id, agent_id, session_id)

        enriched_metadata = metadata.copy() if metadata else {}
        enriched_metadata.update(tenant.to_dict())

        memory = await self._get_memory()
        result = await memory.add(
            messages=messages,
            user_id=tenant.id,
            metadata=enriched_metadata,
            **kwargs,
        )

        result["tenant"] = tenant.to_dict()
        return result

    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Search for relevant facts within tenant scope.

        Args:
            query: Search query
            user_id: User identifier
            agent_id: Agent identifier
            session_id: Session identifier
            limit: Maximum results

        Returns:
            Dict with 'results' containing matching facts
        """
        tenant = self._build_tenant(user_id, agent_id, session_id)

        memory = await self._get_memory()
        result = await memory.search(
            query=query,
            user_id=tenant.id,
            limit=limit,
            **kwargs,
        )

        result["tenant"] = tenant.to_dict()
        return result

    async def get_all_entities(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get all entities for a tenant."""
        tenant = self._build_tenant(user_id, agent_id, session_id)

        memory = await self._get_memory()
        result = await memory.get_all_entities(user_id=tenant.id, limit=limit)

        result["tenant"] = tenant.to_dict()
        return result

    async def get_all_relationships(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get all relationships for a tenant."""
        tenant = self._build_tenant(user_id, agent_id, session_id)

        memory = await self._get_memory()
        result = await memory.get_all_relationships(user_id=tenant.id, limit=limit)

        result["tenant"] = tenant.to_dict()
        return result

    async def delete_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete all graph data for a tenant."""
        tenant = self._build_tenant(user_id, agent_id, session_id)

        memory = await self._get_memory()
        result = await memory.delete_all(user_id=tenant.id)

        result["tenant"] = tenant.to_dict()
        return result

    async def close(self) -> None:
        """Close all connections."""
        if self._graph_memory:
            await self._graph_memory.close()
            self._graph_memory = None

