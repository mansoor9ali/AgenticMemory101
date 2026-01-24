from typing import List, Dict, Any
import logging
from agmem.processing.base import BaseProcessor
from agmem.models import MemoryInput
from agmem.memory.prompts import EVENT_DETECTION_PROMPT
from agmem.memory.utils import extract_json

logger = logging.getLogger(__name__)

class EventDetectionProcessor(BaseProcessor):
    """
    Processor that detects significant events/milestones in input.
    """
    
    def __init__(self, llm_service):
        self.llm = llm_service
        
    async def process(self, input_data: MemoryInput) -> List[MemoryInput]:
        """
        Detect events and create derived event memories.
        """
        if len(input_data.content) < 20:
             return [input_data]
             
        event_data = await self._detect_events(input_data.content)
        
        derived_memories = [input_data]
        
        if event_data and "events" in event_data:
            for event in event_data["events"]:
                summary = f"Event Detected: {event.get('type')} - {event.get('description')}"
                if event.get('date'):
                    summary += f" on {event['date']}"
                    
                derived_memories.append(
                    MemoryInput(
                        content=summary,
                        metadata={
                            "type": "event",
                            "raw_event": event
                        },
                        # Events are salient
                        importance=0.7,
                        # Decay depends on event type, but let's say average
                        metadata_decay=0.01,
                        source="event_detection",
                        tags=["event", event.get("type", "misc").lower()]
                    )
                )
                
        return derived_memories

    async def _detect_events(self, text: str) -> Dict[str, Any]:
        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": EVENT_DETECTION_PROMPT},
                    {"role": "user", "content": text},
                ]
            )
            content = response.get("content", "")
            return extract_json(content) or {}
        except Exception as e:
            logger.warning(f"Event detection failed: {e}")
            return {}
