from abc import ABC, abstractmethod
from typing import List, Any, Dict
from agmem.models import MemoryInput

class BaseProcessor(ABC):
    """
    Abstract base class for memory processors.
    Processors sit in a pipeline and transform/expand inputs.
    """
    
    @abstractmethod
    async def process(self, input_data: MemoryInput) -> List[MemoryInput]:
        """
        Process a single memory input and return a list of derived inputs.
        
        Args:
            input_data: The initial raw memory input.
            
        Returns:
            List[MemoryInput]: A list of processed/derived memory inputs.
                               Return [input_data] to pass through unchanged.
                               Return [] to filter out.
                               Return [input_data, derived_1, derived_2] to expand.
        """
        pass
