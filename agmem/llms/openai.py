"""OpenAI LLM provider."""

import json
import os
import logging
from typing import Any, Dict, List, Optional

from agmem.llms.base import LLMBase
from agmem.exceptions import LLMError

logger = logging.getLogger(__name__)


class OpenAILLM(LLMBase):
    """OpenAI LLM implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        try:
            from openai import OpenAI
        except ImportError:
            raise LLMError(
                "OpenAI library not installed",
                error_code="LLM_OPENAI_001",
                suggestion="Install with: pip install openai",
            )
        
        self._model = self.config.get("model", "gpt-4o-mini")
        self._temperature = self.config.get("temperature", 0.1)
        self._max_tokens = self.config.get("max_tokens", 2000)
        
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = self.config.get("base_url") or os.getenv("OPENAI_BASE_URL")
        
        if not api_key:
            raise LLMError(
                "OpenAI API key not provided",
                error_code="LLM_OPENAI_002",
                suggestion="Set OPENAI_API_KEY environment variable or pass api_key in config",
            )
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate response using OpenAI."""
        try:
            params = {
                "model": self._model,
                "messages": messages,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            }
            
            if tools:
                params["tools"] = tools
                params["tool_choice"] = tool_choice
            
            response = self.client.chat.completions.create(**params)
            
            result = {
                "content": response.choices[0].message.content or "",
                "tool_calls": [],
            }
            
            # Parse tool calls if present
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    result["tool_calls"].append({
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments),
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI LLM call failed: {e}")
            raise LLMError(
                f"Failed to generate response: {e}",
                error_code="LLM_OPENAI_003",
                details={"model": self._model},
            )
    
    @property
    def model_name(self) -> str:
        return self._model
