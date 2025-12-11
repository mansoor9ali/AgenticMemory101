"""Embedding configuration for AgenticMemory."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class OpenAIEmbeddingConfig(BaseModel):
    """Configuration for OpenAI embeddings."""
    
    api_key: Optional[str] = Field(None, description="OpenAI API key (or use env var)")
    model: str = Field("text-embedding-3-small", description="Embedding model name")
    dimensions: int = Field(1536, description="Embedding dimensions")
    base_url: Optional[str] = Field(None, description="Custom base URL")
    
    class Config:
        extra = "forbid"


class CohereEmbeddingConfig(BaseModel):
    """Configuration for Cohere embeddings."""
    
    api_key: Optional[str] = Field(None, description="Cohere API key (or use env var)")
    model: str = Field("embed-english-v3.0", description="Embedding model name")
    
    class Config:
        extra = "forbid"


class OllamaEmbeddingConfig(BaseModel):
    """Configuration for Ollama embeddings (local)."""
    
    base_url: str = Field("http://localhost:11434", description="Ollama server URL")
    model: str = Field("nomic-embed-text", description="Embedding model name")
    
    class Config:
        extra = "forbid"


class HuggingFaceEmbeddingConfig(BaseModel):
    """Configuration for HuggingFace embeddings (local)."""
    
    model: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="Model name")
    device: str = Field("cpu", description="Device to run on")
    
    class Config:
        extra = "forbid"


class EmbedderConfig(BaseModel):
    """Configuration for embedding provider."""
    
    provider: str = Field(
        "openai",
        description="Embedding provider: 'openai', 'cohere', 'ollama', 'huggingface'"
    )
    config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Provider-specific configuration"
    )
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        valid_providers = ["openai", "gemini", "google", "cohere", "ollama", "huggingface", "voyage", "azure_openai"]
        if v not in valid_providers:
            raise ValueError(f"Invalid embedding provider: {v}. Must be one of {valid_providers}")
        return v
    
    class Config:
        extra = "forbid"
