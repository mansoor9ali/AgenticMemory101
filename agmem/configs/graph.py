"""
Configuration for Graph Memory.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from agmem.configs.embeddings import EmbedderConfig
from agmem.configs.llms import LLMConfig


class GraphStoreConfig(BaseModel):
    """Graph database configuration."""
    
    provider: str = Field("neo4j", description="Graph store provider: neo4j")
    uri: str = Field("bolt://localhost:7687", description="Database URI")
    user: str = Field("neo4j", description="Database user")
    password: str = Field("", description="Database password")
    database: str = Field("neo4j", description="Database name")
    
    # Connection pool
    max_connection_pool_size: int = Field(50, description="Max connections in pool")
    connection_timeout: float = Field(30.0, description="Connection timeout in seconds")


class GraphConfig(BaseModel):
    """Main configuration for GraphMemory."""
    
    # Graph store
    graph_store: GraphStoreConfig = Field(
        default_factory=GraphStoreConfig,
        description="Graph database configuration",
    )
    
    # Embeddings (reuse existing)
    embedder: EmbedderConfig = Field(
        default_factory=EmbedderConfig,
        description="Embedding model configuration",
    )
    
    # LLM for extraction (reuse existing)
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM configuration for entity extraction",
    )
    
    # Extraction settings
    max_entities_per_message: int = Field(
        10,
        ge=1,
        le=50,
        description="Maximum entities to extract per message",
    )
    max_relationships_per_message: int = Field(
        15,
        ge=1,
        le=50,
        description="Maximum relationships to extract per message",
    )
    
    # Deduplication
    entity_dedup_threshold: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for entity deduplication",
    )
    
    # Search
    default_search_limit: int = Field(
        10,
        ge=1,
        le=100,
        description="Default number of results to return",
    )
    max_traversal_hops: int = Field(
        3,
        ge=1,
        le=10,
        description="Maximum hops for graph traversal",
    )
    
    # Store raw content
    store_episode_content: bool = Field(
        True,
        description="Whether to store raw episode content",
    )
    
    class Config:
        extra = "forbid"
