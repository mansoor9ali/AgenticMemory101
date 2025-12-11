"""Vector store configuration for AgenticMemory."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class PineconeConfig(BaseModel):
    """Configuration for Pinecone vector store."""
    
    api_key: str = Field(..., description="Pinecone API key")
    index_name: str = Field("agmem", description="Index name")
    environment: Optional[str] = Field(None, description="Pinecone environment")
    namespace: Optional[str] = Field(None, description="Namespace for isolation")
    
    class Config:
        extra = "forbid"


class QdrantConfig(BaseModel):
    """Configuration for Qdrant vector store."""
    
    url: str = Field("http://localhost:6333", description="Qdrant server URL")
    api_key: Optional[str] = Field(None, description="Qdrant API key")
    collection_name: str = Field("agmem", description="Collection name")
    
    class Config:
        extra = "forbid"


class ChromaConfig(BaseModel):
    """Configuration for Chroma vector store (local)."""
    
    path: str = Field("~/.agmem/chroma", description="Path to Chroma database")
    collection_name: str = Field("agmem", description="Collection name")
    
    class Config:
        extra = "forbid"


class PGVectorConfig(BaseModel):
    """Configuration for pgvector (PostgreSQL extension)."""
    
    host: str = Field("localhost", description="Database host")
    port: int = Field(5432, description="Database port")
    database: str = Field("agmem", description="Database name")
    user: str = Field(..., description="Database user")
    password: str = Field(..., description="Database password")
    collection_name: str = Field("memory_vectors", description="Table name")
    embedding_dims: int = Field(1536, description="Embedding dimensions")
    
    class Config:
        extra = "forbid"


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    
    provider: str = Field(
        "chroma",
        description="Vector store provider: 'pinecone', 'qdrant', 'chroma', 'pgvector'"
    )
    config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Provider-specific configuration"
    )
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        valid_providers = ["pinecone", "qdrant", "chroma", "pgvector", "weaviate", "milvus"]
        if v not in valid_providers:
            raise ValueError(f"Invalid vector store provider: {v}. Must be one of {valid_providers}")
        return v
    
    class Config:
        extra = "forbid"
