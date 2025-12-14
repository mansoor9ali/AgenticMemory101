"""
Configuration for Graph Memory.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from agmem.configs.embeddings import EmbedderConfig
from agmem.configs.llms import LLMConfig


class GraphStoreConfig(BaseModel):
    """Graph database configuration."""
    
    provider: str = Field(
        "neo4j",
        description="Graph store provider: neo4j or falkordb",
    )
    
    # Neo4j connection settings
    uri: str = Field("bolt://localhost:7687", description="Neo4j database URI")
    user: str = Field("neo4j", description="Neo4j/FalkorDB username")
    password: str = Field("", description="Database password")
    database: str = Field("neo4j", description="Neo4j database name")
    max_connection_pool_size: int = Field(50, description="Neo4j max connections in pool")
    connection_timeout: float = Field(30.0, description="Neo4j connection timeout in seconds")
    
    # FalkorDB connection settings
    host: str = Field("localhost", description="FalkorDB host (when provider=falkordb)")
    port: int = Field(6379, description="FalkorDB port (when provider=falkordb)")
    graph_name: str = Field(
        "agentic_memory",
        description="Graph key/name inside FalkorDB/RedisGraph",
    )
    use_tls: bool = Field(False, description="Use TLS when connecting to FalkorDB")


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
