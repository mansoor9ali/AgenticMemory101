"""Factory classes for creating providers."""

import logging
from typing import Any, Dict, Optional

from agmem.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class EmbedderFactory:
    """Factory for creating embedding providers."""
    
    provider_map = {
        "openai": "agmem.embeddings.openai.OpenAIEmbedding",
        "gemini": "agmem.embeddings.gemini.GeminiEmbedding",
        "google": "agmem.embeddings.gemini.GeminiEmbedding",
        "cohere": "agmem.embeddings.cohere.CohereEmbedding",
        "ollama": "agmem.embeddings.ollama.OllamaEmbedding",
        "huggingface": "agmem.embeddings.huggingface.HuggingFaceEmbedding",
        "azure_openai": "agmem.embeddings.azure_openai.AzureOpenAIEmbedding",
    }
    
    @classmethod
    def create(cls, provider: str, config: Optional[Dict[str, Any]] = None):
        """Create an embedding provider instance."""
        if provider not in cls.provider_map:
            raise ConfigurationError(
                f"Unknown embedding provider: {provider}",
                error_code="CFG_EMBED_001",
                details={"provider": provider, "available": list(cls.provider_map.keys())},
            )
        
        module_path = cls.provider_map[provider]
        module_name, class_name = module_path.rsplit(".", 1)
        
        try:
            import importlib
            module = importlib.import_module(module_name)
            provider_class = getattr(module, class_name)
            return provider_class(config or {})
        except ImportError as e:
            raise ConfigurationError(
                f"Failed to import embedding provider '{provider}': {e}",
                error_code="CFG_EMBED_002",
                suggestion=f"Install the required dependencies for {provider}",
            )


class StorageFactory:
    """Factory for creating L2 storage providers."""
    
    provider_map = {
        "postgres": "agmem.storage.postgres.PostgresStorage",
        "postgresql": "agmem.storage.postgres.PostgresStorage",
        "mongodb": "agmem.storage.mongodb.MongoDBStorage",
        "mongo": "agmem.storage.mongodb.MongoDBStorage",
        "sqlite": "agmem.storage.sqlite.SQLiteStorage",
    }
    
    @classmethod
    def create(cls, provider: str, config: Optional[Dict[str, Any]] = None):
        """Create a storage provider instance."""
        if provider not in cls.provider_map:
            raise ConfigurationError(
                f"Unknown storage provider: {provider}",
                error_code="CFG_STORAGE_001",
                details={"provider": provider, "available": list(cls.provider_map.keys())},
            )
        
        module_path = cls.provider_map[provider]
        module_name, class_name = module_path.rsplit(".", 1)
        
        try:
            import importlib
            module = importlib.import_module(module_name)
            provider_class = getattr(module, class_name)
            return provider_class(config or {})
        except ImportError as e:
            raise ConfigurationError(
                f"Failed to import storage provider '{provider}': {e}",
                error_code="CFG_STORAGE_002",
                suggestion=f"Install the required dependencies for {provider}",
            )


class VectorStoreFactory:
    """Factory for creating vector store providers."""
    
    provider_map = {
        "pinecone": "agmem.vector_stores.pinecone.PineconeVectorStore",
        "qdrant": "agmem.vector_stores.qdrant.QdrantVectorStore",
        "chroma": "agmem.vector_stores.chroma.ChromaVectorStore",
        "chromadb": "agmem.vector_stores.chroma.ChromaVectorStore",
    }
    
    @classmethod
    def create(cls, provider: str, config: Optional[Dict[str, Any]] = None):
        """Create a vector store provider instance."""
        if provider not in cls.provider_map:
            raise ConfigurationError(
                f"Unknown vector store provider: {provider}",
                error_code="CFG_VECTOR_001",
                details={"provider": provider, "available": list(cls.provider_map.keys())},
            )
        
        module_path = cls.provider_map[provider]
        module_name, class_name = module_path.rsplit(".", 1)
        
        try:
            import importlib
            module = importlib.import_module(module_name)
            provider_class = getattr(module, class_name)
            return provider_class(config or {})
        except ImportError as e:
            raise ConfigurationError(
                f"Failed to import vector store provider '{provider}': {e}",
                error_code="CFG_VECTOR_002",
                suggestion=f"Install the required dependencies for {provider}",
            )


class LLMFactory:
    """Factory for creating LLM providers."""
    
    provider_map = {
        "openai": "agmem.llms.openai.OpenAILLM",
        "gemini": "agmem.llms.gemini.GeminiLLM",
        "google": "agmem.llms.gemini.GeminiLLM",
        "anthropic": "agmem.llms.anthropic.AnthropicLLM",
        "ollama": "agmem.llms.ollama.OllamaLLM",
    }
    
    @classmethod
    def create(cls, provider: str, config: Optional[Dict[str, Any]] = None):
        """Create an LLM provider instance."""
        if provider not in cls.provider_map:
            raise ConfigurationError(
                f"Unknown LLM provider: {provider}",
                error_code="CFG_LLM_001",
                details={"provider": provider, "available": list(cls.provider_map.keys())},
            )
        
        module_path = cls.provider_map[provider]
        module_name, class_name = module_path.rsplit(".", 1)
        
        try:
            import importlib
            module = importlib.import_module(module_name)
            provider_class = getattr(module, class_name)
            return provider_class(config or {})
        except ImportError as e:
            raise ConfigurationError(
                f"Failed to import LLM provider '{provider}': {e}",
                error_code="CFG_LLM_002",
                suggestion=f"Install the required dependencies for {provider}",
            )
