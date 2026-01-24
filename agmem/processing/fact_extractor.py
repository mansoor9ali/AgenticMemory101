from typing import List, Dict, Any, Optional
import logging
from agmem.processing.base import BaseProcessor
from agmem.models import MemoryInput
from agmem.memory.prompts import FACT_EXTRACTION_PROMPT
from agmem.memory.utils import extract_json

logger = logging.getLogger(__name__)

class FactExtractionProcessor(BaseProcessor):
    """
    Processor that uses an LLM to extract atomic facts from raw input.
    """
    
    def __init__(self, llm_service):
        """
        Args:
            llm_service: The initialized LLM service to use.
        """
        self.llm = llm_service
        
    async def process(self, input_data: MemoryInput) -> List[MemoryInput]:
        """
        Extract facts from the input content.
        
        If the input source is already 'extraction' or 'system', we might skip.
        For now, we process 'manual' and 'conversation' sources.
        """
        
        # Skip if already processed or not suitable for extraction
        if input_data.source in ["extraction", "profile_update"]:
            return [input_data]
            
        facts_data = await self._extract_facts(input_data.content)
        
        if not facts_data:
            # Fallback: maintain original if no facts extracted (or maybe we want to drop it?)
            # Let's keep original if nothing extracted, or better yet, return nothing 
            # if we strictly want atomic facts. 
            # DECISION: Return the original as a backup if no facts found, 
            # BUT if facts found, return ONLY the facts (decomposition).
            return [input_data]
            
        derived_memories = []
        for fact in facts_data:
             derived_memories.append(
                 MemoryInput(
                     content=fact["text"],
                     # Merge metadata with extracted importance/decay
                     metadata={**input_data.metadata},
                     # Use extracted importance if available
                     importance=fact.get("importance", input_data.importance),
                     source="extraction",
                     tags=input_data.tags + ["extracted"]
                 )
             )
             
        return derived_memories

    async def _extract_facts(self, text: str) -> List[Dict[str, Any]]:
        """
        Use LLM to split text into atomic facts.
        """
        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": FACT_EXTRACTION_PROMPT},
                    {"role": "user", "content": text},
                ]
            )
            
            content = response.get("content", "")
            parsed = extract_json(content)
            
            if parsed and "facts" in parsed:
                return self._normalize_facts(parsed["facts"])
                
            return []
            
        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
            return []

    def _normalize_facts(self, facts: List[Any]) -> List[Dict[str, Any]]:
        normalized = []
        for fact in facts:
            if isinstance(fact, str):
                normalized.append({
                    "text": fact,
                    "importance": 0.5,
                    "decay": 0.01
                })
            elif isinstance(fact, dict) and "text" in fact:
                normalized.append({
                    "text": fact["text"],
                    "importance": float(fact.get("importance", 0.5)),
                    "decay": float(fact.get("decay", 0.01))
                })
        return normalized
