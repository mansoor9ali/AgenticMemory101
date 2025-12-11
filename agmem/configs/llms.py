"""LLM configuration for AgenticMemory."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class OpenAILLMConfig(BaseModel):
    """Configuration for OpenAI LLM."""
    
    api_key: Optional[str] = Field(None, description="OpenAI API key (or use env var)")
    model: str = Field("gpt-4o-mini", description="Model name")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="Temperature")
    max_tokens: int = Field(2000, ge=1, description="Max tokens")
    base_url: Optional[str] = Field(None, description="Custom base URL")
    
    class Config:
        extra = "forbid"


class AnthropicLLMConfig(BaseModel):
    """Configuration for Anthropic LLM."""
    
    api_key: Optional[str] = Field(None, description="Anthropic API key (or use env var)")
    model: str = Field("claude-3-5-sonnet-20240620", description="Model name")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="Temperature")
    max_tokens: int = Field(2000, ge=1, description="Max tokens")
    
    class Config:
        extra = "forbid"


class OllamaLLMConfig(BaseModel):
    """Configuration for Ollama LLM (local)."""
    
    base_url: str = Field("http://localhost:11434", description="Ollama server URL")
    model: str = Field("llama3.2", description="Model name")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="Temperature")
    
    class Config:
        extra = "forbid"


class LLMConfig(BaseModel):
    """Configuration for LLM provider."""
    
    provider: str = Field(
        "openai",
        description="LLM provider: 'openai', 'anthropic', 'ollama'"
    )
    config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Provider-specific configuration"
    )
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        valid_providers = ["openai", "gemini", "google", "anthropic", "ollama", "azure_openai", "bedrock"]
        if v not in valid_providers:
            raise ValueError(f"Invalid LLM provider: {v}. Must be one of {valid_providers}")
        return v
    
    class Config:
        extra = "forbid"
