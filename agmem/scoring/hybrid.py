from datetime import datetime, timezone
from agmem.scoring.base import BaseScorer
from agmem.models import Memory
from agmem.memory.utils import compute_importance, compute_keyword_overlap

class HybridScorer(BaseScorer):
    """
    Default hybrid scorer combining:
    - Semantic Similarity (Vector)
    - Recency & Importance (Memory Lifecycle)
    - Keyword Overlap (Lexical)
    """
    
    def __init__(self, weight_semantic=0.5, weight_importance=0.3, weight_keyword=0.2):
        self.w_sem = weight_semantic
        self.w_imp = weight_importance
        self.w_key = weight_keyword
        
    def score(
        self, 
        query: str, 
        memory: Memory, 
        semantic_score: float,
        context: dict = None
    ) -> float:
        
        now = datetime.now(tz=timezone.utc)
        
        # Calculate Importance/Recency Score
        # Reuse existing util which already handles recency decay
        lifecycle_score = compute_importance(
            memory,
            now,
            # We assume these weights are handled inside compute_importance logic or passed here
            # For checking, `compute_importance` takes (memory, now, w_recency, w_frequency, w_importance)
            # We'll use defaults or config values if passed in context, otherwise defaults
            weight_recency=0.5,
            weight_frequency=0.1,
            weight_importance=0.4
        )
        
        # Keyword Score
        keyword_score = compute_keyword_overlap(query, memory.content)
        
        # Weighted Final Score
        final_score = (
            self.w_sem * semantic_score +
            self.w_imp * lifecycle_score +
            self.w_key * keyword_score
        )
        
        return final_score
