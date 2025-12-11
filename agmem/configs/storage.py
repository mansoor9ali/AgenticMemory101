"""Storage configuration for AgenticMemory."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class PostgresConfig(BaseModel):
    """Configuration for PostgreSQL storage."""
    
    host: str = Field("localhost", description="Database host")
    port: int = Field(5432, description="Database port")
    database: str = Field("agmem", description="Database name")
    user: str = Field(..., description="Database user")
    password: str = Field(..., description="Database password")
    collection_name: str = Field("memories", description="Table name")
    min_connections: int = Field(1, ge=1, description="Minimum pool connections")
    max_connections: int = Field(10, ge=1, description="Maximum pool connections")
    
    class Config:
        extra = "forbid"


class MongoDBConfig(BaseModel):
    """Configuration for MongoDB storage."""
    
    uri: str = Field("mongodb://localhost:27017", description="MongoDB connection URI")
    database: str = Field("agmem", description="Database name")
    collection_name: str = Field("memories", description="Collection name")
    
    class Config:
        extra = "forbid"


class SQLiteConfig(BaseModel):
    """Configuration for SQLite storage (local/dev)."""
    
    path: str = Field("~/.agmem/memories.db", description="Path to SQLite database")
    collection_name: str = Field("memories", description="Table name")
    
    class Config:
        extra = "forbid"


class StorageConfig(BaseModel):
    """Configuration for L2 storage."""
    
    provider: str = Field(
        "sqlite",
        description="Storage provider: 'postgres', 'mongodb', 'sqlite'"
    )
    config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Provider-specific configuration"
    )
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        valid_providers = ["postgres", "mongodb", "sqlite", "dynamodb"]
        if v not in valid_providers:
            raise ValueError(f"Invalid storage provider: {v}. Must be one of {valid_providers}")
        return v
    
    class Config:
        extra = "forbid"
