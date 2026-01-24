from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from agmem.models import Memory

class BaseScorer(ABC):
    """
    Abstract base class for memory scoring strategies.
    """
    
    @abstractmethod
    def score(
        self, 
        query: str, 
        memory: Memory, 
        semantic_score: float,
        context: Dict[str, Any] = None
    ) -> float:
        """
        Calculate a relevance score for a memory given a query.
        
        Args:
           query: The search query string.
           memory: The memory object candidate.
           semantic_score: The raw vector similarity score [0..1] provided by vector store.
           context: Optional context (current time, location, etc.)
           
        Returns:
           float: A final score [0..1] used for ranking.
        """
        pass
