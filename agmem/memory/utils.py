"""Utility functions for memory operations."""

import json
import logging
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from agmem.models import Memory

logger = logging.getLogger(__name__)


def compute_importance(
    memory: Memory,
    now: datetime,
    weight_recency: float = 0.4,
    weight_frequency: float = 0.3,
    weight_importance: float = 0.3,
) -> float:
    """
    Compute the current importance score of a memory.
    
    Combines:
    - Recency: exponential decay based on last access time
    - Frequency: log-scaled access count
    - Base importance: user-set or extracted importance
    """
    # Recency factor (exponential decay)
    # Handle timezone-aware vs naive datetime comparison
    last_accessed = memory.last_accessed_at
    if now.tzinfo is not None and last_accessed.tzinfo is None:
        # now is aware, last_accessed is naive - assume UTC
        from datetime import timezone
        last_accessed = last_accessed.replace(tzinfo=timezone.utc)
    elif now.tzinfo is None and last_accessed.tzinfo is not None:
        # now is naive, last_accessed is aware - make now UTC
        from datetime import timezone
        now = now.replace(tzinfo=timezone.utc)
    
    age_hours = (now - last_accessed).total_seconds() / 3600
    recency = math.exp(-memory.decay_rate * age_hours)
    
    # Frequency factor (log scale to prevent dominance)
    frequency = min(1.0, math.log(1 + memory.access_count) / 10)
    
    # Base importance
    base = memory.importance
    
    # Weighted combination
    score = (
        weight_recency * recency +
        weight_frequency * frequency +
        weight_importance * base
    )
    
    return min(1.0, max(0.0, score))


def compute_keyword_overlap(query: str, content: str) -> float:
    """Simple keyword overlap score between query and content."""
    query_words = set(query.lower().split())
    content_words = set(content.lower().split())
    
    if not query_words:
        return 0.0
    
    overlap = len(query_words & content_words)
    return overlap / len(query_words)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def is_duplicate(
    new_embedding: List[float],
    existing_memories: List[Memory],
    threshold: float = 0.9,
) -> bool:
    """Check if content is a duplicate of existing memories."""
    for memory in existing_memories:
        if memory.embedding:
            similarity = cosine_similarity(new_embedding, memory.embedding)
            if similarity >= threshold:
                return True
    return False


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from text, handling markdown code blocks."""
    # Try to find JSON in code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        text = json_match.group(1)
    
    # Try to parse as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in text
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
    
    logger.warning(f"Failed to extract JSON from: {text[:100]}...")
    return None


def format_memories_for_prompt(memories: List[Memory]) -> str:
    """Format memories for inclusion in prompts."""
    if not memories:
        return "[]"
    
    formatted = []
    for mem in memories:
        formatted.append({
            "id": mem.id,
            "text": mem.content,
        })
    
    return json.dumps(formatted, indent=2)
