"""Google Gemini embedding provider."""

import os
import logging
from typing import Any, Dict, List, Optional

from agmem.embeddings.base import EmbeddingBase
from agmem.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class GeminiEmbedding(EmbeddingBase):
    """Google Gemini embedding implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        try:
            from google import genai
            from google.genai import types
            self._types = types
        except ImportError:
            raise EmbeddingError(
                "google-genai library not installed",
                error_code="EMBED_GEMINI_001",
                suggestion="Install with: pip install google-genai",
            )
        
        self._model = self.config.get("model", "models/text-embedding-004")
        self._dimensions = self.config.get("dimensions", 768)
        
        api_key = self.config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise EmbeddingError(
                "Google API key not provided",
                error_code="EMBED_GEMINI_002",
                suggestion="Set GOOGLE_API_KEY environment variable or pass api_key in config",
            )
        
        self.client = genai.Client(api_key=api_key)
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text using Gemini."""
        try:
            text = text.replace("\n", " ")
            
            config = self._types.EmbedContentConfig(
                output_dimensionality=self._dimensions
            )
            
            response = self.client.models.embed_content(
                model=self._model,
                contents=text,
                config=config,
            )
            
            return response.embeddings[0].values
            
        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            raise EmbeddingError(
                f"Failed to generate embedding: {e}",
                error_code="EMBED_GEMINI_003",
                details={"model": self._model, "text_length": len(text)},
            )
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            texts = [t.replace("\n", " ") for t in texts]
            
            config = self._types.EmbedContentConfig(
                output_dimensionality=self._dimensions
            )
            
            # Gemini supports batch embedding
            results = []
            for text in texts:
                response = self.client.models.embed_content(
                    model=self._model,
                    contents=text,
                    config=config,
                )
                results.append(response.embeddings[0].values)
            
            return results
            
        except Exception as e:
            logger.error(f"Gemini batch embedding failed: {e}")
            raise EmbeddingError(
                f"Failed to generate batch embeddings: {e}",
                error_code="EMBED_GEMINI_004",
                details={"model": self._model, "batch_size": len(texts)},
            )
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    @property
    def model_name(self) -> str:
        return self._model
