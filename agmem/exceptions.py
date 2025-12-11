"""
Structured exception classes for AgenticMemory with error codes and debug information.
"""

from typing import Any, Dict, Optional


class AgenticMemoryError(Exception):
    """Base exception for all AgenticMemory errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "REMINA_001",
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        debug_info: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.suggestion = suggestion
        self.debug_info = debug_info or {}
        super().__init__(self.message)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"error_code={self.error_code!r})"
        )


class ConfigurationError(AgenticMemoryError):
    """Raised when configuration is invalid."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "CFG_001",
        details: Optional[Dict[str, Any]] = None,
        suggestion: str = "Please check your configuration settings",
        debug_info: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details, suggestion, debug_info)


class StorageError(AgenticMemoryError):
    """Raised when L2 storage operations fail."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "STORAGE_001",
        details: Optional[Dict[str, Any]] = None,
        suggestion: str = "Please check your storage configuration and connection",
        debug_info: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details, suggestion, debug_info)


class VectorStoreError(AgenticMemoryError):
    """Raised when vector store operations fail."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "VECTOR_001",
        details: Optional[Dict[str, Any]] = None,
        suggestion: str = "Please check your vector store configuration",
        debug_info: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details, suggestion, debug_info)


class EmbeddingError(AgenticMemoryError):
    """Raised when embedding operations fail."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "EMBED_001",
        details: Optional[Dict[str, Any]] = None,
        suggestion: str = "Please check your embedding model configuration",
        debug_info: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details, suggestion, debug_info)


class LLMError(AgenticMemoryError):
    """Raised when LLM operations fail."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "LLM_001",
        details: Optional[Dict[str, Any]] = None,
        suggestion: str = "Please check your LLM configuration and API key",
        debug_info: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details, suggestion, debug_info)


class CacheError(AgenticMemoryError):
    """Raised when cache operations fail."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "CACHE_001",
        details: Optional[Dict[str, Any]] = None,
        suggestion: str = "Cache operation failed, will continue without cache",
        debug_info: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details, suggestion, debug_info)


class MemoryNotFoundError(AgenticMemoryError):
    """Raised when a memory is not found."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "MEM_404",
        details: Optional[Dict[str, Any]] = None,
        suggestion: str = "Please check the memory ID and ensure it exists",
        debug_info: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details, suggestion, debug_info)


class RateLimitError(AgenticMemoryError):
    """Raised when rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "RATE_001",
        details: Optional[Dict[str, Any]] = None,
        suggestion: str = "Please wait before making more requests",
        debug_info: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details, suggestion, debug_info)
