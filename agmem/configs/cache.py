"""Cache configuration for AgenticMemory."""

from typing import Optional
from pydantic import BaseModel, Field


class CacheConfig(BaseModel):
    """Configuration for Redis L1 cache."""
    
    redis_url: str = Field(
        "redis://localhost:6379",
        description="Redis connection URL"
    )
    ttl_seconds: int = Field(
        3600,
        ge=60,
        description="TTL for cached memories in seconds"
    )
    max_memories_per_user: int = Field(
        100,
        ge=10,
        description="Maximum memories to cache per user"
    )
    enabled: bool = Field(
        True,
        description="Whether to enable L1 caching"
    )
    key_prefix: str = Field(
        "agmem",
        description="Prefix for Redis keys"
    )
    
    class Config:
        extra = "forbid"
