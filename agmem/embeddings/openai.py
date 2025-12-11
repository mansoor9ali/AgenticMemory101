"""OpenAI embedding provider."""

import os
import logging
from typing import Any, Dict, List, Optional

from agmem.embeddings.base import EmbeddingBase
from agmem.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class OpenAIEmbedding(EmbeddingBase):
    """OpenAI embedding implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        try:
            from openai import OpenAI
        except ImportError:
            raise EmbeddingError(
                "OpenAI library not installed",
                error_code="EMBED_OPENAI_001",
                suggestion="Install with: pip install openai",
            )
        
        self._model = self.config.get("model", "text-embedding-3-small")
        self._dimensions = self.config.get("dimensions", 1536)
        
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = self.config.get("base_url") or os.getenv("OPENAI_BASE_URL")
        
        if not api_key:
            raise EmbeddingError(
                "OpenAI API key not provided",
                error_code="EMBED_OPENAI_002",
                suggestion="Set OPENAI_API_KEY environment variable or pass api_key in config",
            )
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        try:
            text = text.replace("\n", " ")
            response = self.client.embeddings.create(
                input=[text],
                model=self._model,
                dimensions=self._dimensions,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise EmbeddingError(
                f"Failed to generate embedding: {e}",
                error_code="EMBED_OPENAI_003",
                details={"model": self._model, "text_length": len(text)},
            )
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            texts = [t.replace("\n", " ") for t in texts]
            response = self.client.embeddings.create(
                input=texts,
                model=self._model,
                dimensions=self._dimensions,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI batch embedding failed: {e}")
            raise EmbeddingError(
                f"Failed to generate batch embeddings: {e}",
                error_code="EMBED_OPENAI_004",
                details={"model": self._model, "batch_size": len(texts)},
            )
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    @property
    def model_name(self) -> str:
        return self._model
