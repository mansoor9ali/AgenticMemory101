"""Base configuration for AgenticMemory Memory."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from agmem.configs.storage import StorageConfig
from agmem.configs.vectors import VectorStoreConfig
from agmem.configs.embeddings import EmbedderConfig
from agmem.configs.llms import LLMConfig
from agmem.configs.cache import CacheConfig


class MemoryItem(BaseModel):
    """Represents a memory item returned from search/get operations."""
    
    id: str = Field(..., description="Unique identifier for the memory")
    memory: str = Field(..., description="The memory content")
    hash: Optional[str] = Field(None, description="Hash of the memory content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    score: Optional[float] = Field(None, description="Relevance score")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


class MemoryConfig(BaseModel):
    """Main configuration for AgenticMemory Memory."""
    
    # L1 Cache (Redis - fixed)
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Redis L1 cache configuration"
    )
    
    # L2 Storage (pluggable)
    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        description="Long-term storage configuration"
    )
    
    # Vector Store (pluggable)
    vector_store: VectorStoreConfig = Field(
        default_factory=VectorStoreConfig,
        description="Vector store configuration"
    )
    
    # Embeddings (pluggable)
    embedder: EmbedderConfig = Field(
        default_factory=EmbedderConfig,
        description="Embedding model configuration"
    )
    
    # LLM (pluggable)
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM configuration for extraction"
    )
    
    # Scoring weights
    weight_recency: float = Field(0.4, ge=0.0, le=1.0, description="Weight for recency in scoring")
    weight_frequency: float = Field(0.3, ge=0.0, le=1.0, description="Weight for frequency in scoring")
    weight_importance: float = Field(0.3, ge=0.0, le=1.0, description="Weight for importance in scoring")
    
    # Consolidation
    consolidation_threshold: float = Field(0.85, ge=0.0, le=1.0, description="Similarity threshold for consolidation")
    
    # Deduplication
    dedup_threshold: float = Field(0.9, ge=0.0, le=1.0, description="Similarity threshold for deduplication")
    
    # Extraction
    max_facts_per_conversation: int = Field(10, ge=1, description="Max facts to extract per conversation")
    
    # Custom prompts
    custom_extraction_prompt: Optional[str] = Field(None, description="Custom prompt for fact extraction")
    
    class Config:
        extra = "forbid"
