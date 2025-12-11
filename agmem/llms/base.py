"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMBase(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            tool_choice: How to handle tool selection
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Response dict with 'content' and optionally 'tool_calls'
        """
        pass
    
    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.config.get("model", "unknown")
