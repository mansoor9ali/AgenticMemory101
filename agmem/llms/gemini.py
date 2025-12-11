"""Google Gemini LLM provider."""

import json
import os
import logging
from typing import Any, Dict, List, Optional

from agmem.llms.base import LLMBase
from agmem.exceptions import LLMError

logger = logging.getLogger(__name__)


class GeminiLLM(LLMBase):
    """Google Gemini LLM implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        try:
            from google import genai
            from google.genai import types
            self._types = types
        except ImportError:
            raise LLMError(
                "google-genai library not installed",
                error_code="LLM_GEMINI_001",
                suggestion="Install with: pip install google-genai",
            )
        
        self._model = self.config.get("model", "gemini-2.0-flash")
        self._temperature = self.config.get("temperature", 0.1)
        self._max_tokens = self.config.get("max_tokens", 2000)
        
        api_key = self.config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise LLMError(
                "Google API key not provided",
                error_code="LLM_GEMINI_002",
                suggestion="Set GOOGLE_API_KEY environment variable or pass api_key in config",
            )
        
        self.client = genai.Client(api_key=api_key)
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate response using Gemini."""
        try:
            # Convert messages to Gemini format
            contents = []
            system_instruction = None
            
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    system_instruction = content
                elif role == "user":
                    contents.append(
                        self._types.Content(
                            role="user",
                            parts=[self._types.Part(text=content)]
                        )
                    )
                elif role == "assistant":
                    contents.append(
                        self._types.Content(
                            role="model",
                            parts=[self._types.Part(text=content)]
                        )
                    )
            
            # Build config
            config = self._types.GenerateContentConfig(
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
            )
            
            if system_instruction:
                config.system_instruction = system_instruction
            
            # Generate response
            response = self.client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
            
            result = {
                "content": response.text or "",
                "tool_calls": [],
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini LLM call failed: {e}")
            raise LLMError(
                f"Failed to generate response: {e}",
                error_code="LLM_GEMINI_003",
                details={"model": self._model},
            )
    
    @property
    def model_name(self) -> str:
        return self._model
