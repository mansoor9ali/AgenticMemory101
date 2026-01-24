from typing import List, Dict, Any
import logging
from agmem.processing.base import BaseProcessor
from agmem.models import MemoryInput
from agmem.memory.prompts import USER_PROFILING_PROMPT
from agmem.memory.utils import extract_json

logger = logging.getLogger(__name__)

class UserProfilingProcessor(BaseProcessor):
    """
    Processor that updates a user profile based on observed behavior/text.
    """
    
    def __init__(self, llm_service):
        self.llm = llm_service
        
    async def process(self, input_data: MemoryInput) -> List[MemoryInput]:
        """
        Derive profiling info from input.
        We create a NEW memory for the profile update if significant traits are found.
        """
        # Iterate only if we have substantial content
        if len(input_data.content) < 50: 
            return [input_data]
            
        profile_data = await self._profile_user(input_data.content)
        
        derived_memories = [input_data]
        
        if profile_data:
            # Create a summary string or structured object
            # For this simple implementation, we assume we just store the profile as a memory
            # In a real system, we might update a separate 'User Profile' entity
            
            traits_str = ", ".join(profile_data.get("traits", []))
            values_str = ", ".join(profile_data.get("values", []))
            style = profile_data.get("style", "Unknown")
            
            summary = f"User Profile Update: Style={style}. Traits=[{traits_str}]. Values=[{values_str}]"
            
            if traits_str or values_str:
                derived_memories.append(
                    MemoryInput(
                        content=summary,
                        metadata={
                            "type": "user_profile", 
                            "raw_profile": profile_data
                        },
                        # Profiles are important!
                        importance=0.8,
                        # Profiles don't decay fast
                        metadata_decay=0.005, 
                        source="profile_update",
                        tags=["profile", "auto-generated"]
                    )
                )
                
        return derived_memories

    async def _profile_user(self, text: str) -> Dict[str, Any]:
        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": USER_PROFILING_PROMPT},
                    {"role": "user", "content": text},
                ]
            )
            content = response.get("content", "")
            return extract_json(content) or {}
            
        except Exception as e:
            logger.warning(f"Profiling failed: {e}")
            return {}
