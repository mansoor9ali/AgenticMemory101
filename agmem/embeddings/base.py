"""Base class for embedding providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class EmbeddingBase(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        pass
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Default implementation calls embed() for each text.
        Override for batch optimization.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return [self.embed(text) for text in texts]
    
    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        return self.config.get("dimensions", 1536)
    
    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.config.get("model", "unknown")
