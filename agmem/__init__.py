"""
AgenticMemory - A pluggable memory framework for AI applications.
"""

import importlib.metadata

__version__ = "0.1.0"

from agmem.memory.main import Memory, AsyncMemory
from agmem.exceptions import (
    AgenticMemoryError,
    StorageError,
    VectorStoreError,
    EmbeddingError,
    LLMError,
    CacheError,
    ConfigurationError,
)

# Graph memory (optional - requires neo4j)
try:
    from agmem.graph.main import GraphMemory, AsyncGraphMemory
except ImportError:
    GraphMemory = None
    AsyncGraphMemory = None

__all__ = [
    # Vector memory
    "Memory",
    "AsyncMemory",
    # Graph memory
    "GraphMemory",
    "AsyncGraphMemory",
    # Exceptions
    "AgenticMemoryError",
    "StorageError",
    "VectorStoreError",
    "EmbeddingError",
    "LLMError",
    "CacheError",
    "ConfigurationError",
]
