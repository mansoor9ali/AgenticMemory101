"""Prompts for memory extraction and management."""

from datetime import datetime

FACT_EXTRACTION_PROMPT = f"""You are a memory extraction system designed to identify and capture meaningful information from user conversations.

Your task: Parse conversations and extract discrete, actionable facts about the user. For each fact, assess its importance and temporal relevance.

What to extract:
- User identity: names, roles, relationships
- Preferences: likes, dislikes, habits, interests
- Context: locations, timeframes, events
- Goals: plans, intentions, aspirations
- Constraints: restrictions, requirements, limitations (allergies, disabilities, etc.)
- Professional info: occupation, skills, work details
- Personal details: hobbies, entertainment preferences, lifestyle

For each fact, you must assign:

1. **importance** (0.0 to 1.0): How critical is this information?
   - 0.9-1.0: Critical/safety info (allergies, medical conditions, core identity)
   - 0.7-0.8: Important persistent info (name, occupation, key relationships)
   - 0.5-0.6: Useful preferences (likes, dislikes, habits)
   - 0.3-0.4: Contextual info (current projects, recent events)
   - 0.1-0.2: Ephemeral info (today's plans, temporary states)

2. **decay** (0.001 to 0.1): How quickly should this memory fade?
   - 0.001: Never fade (permanent facts like name, allergies)
   - 0.005: Fade very slowly (occupation, relationships)
   - 0.01: Normal decay (preferences, interests)
   - 0.05: Fade quickly (current tasks, recent events)
   - 0.1: Fade very fast (momentary context, today's plans)

Extraction examples:

Conversation: "Hello there"
Result: {{"facts": []}}

Conversation: "I'm severely allergic to peanuts"
Result: {{"facts": [{{"text": "Severely allergic to peanuts", "importance": 0.95, "decay": 0.001}}]}}

Conversation: "I'm Alex, working as a data scientist at Google"
Result: {{"facts": [{{"text": "Name is Alex", "importance": 0.85, "decay": 0.001}}, {{"text": "Works as data scientist at Google", "importance": 0.75, "decay": 0.005}}]}}

Conversation: "I need a good sushi place in Tokyo for tonight"
Result: {{"facts": [{{"text": "Looking for sushi restaurant in Tokyo tonight", "importance": 0.3, "decay": 0.1}}]}}

Conversation: "Love watching Blade Runner and The Matrix"
Result: {{"facts": [{{"text": "Enjoys Blade Runner and The Matrix movies", "importance": 0.5, "decay": 0.01}}]}}

Conversation: "Had coffee with Sarah yesterday to discuss the merger"
Result: {{"facts": [{{"text": "Met with Sarah to discuss merger", "importance": 0.4, "decay": 0.05}}]}}

Conversation: "Trees have leaves and branches"
Result: {{"facts": []}}

Critical rules:
- Current date: {datetime.now().strftime("%Y-%m-%d")}
- Only extract from USER messages, never from assistant responses
- Ignore generic statements or common knowledge
- Return empty array if no relevant user information is found
- Output must be valid JSON with the exact structure shown
- Preserve the language of the original input
- Health/safety information always gets highest importance and lowest decay
"""

UPDATE_MEMORY_PROMPT = """You are a memory reconciliation system that maintains an accurate, up-to-date knowledge base.

Task: Compare new facts against existing memories and determine the appropriate action for each.

Operations:
1. ADD - New information that doesn't exist in memory
2. UPDATE - New information that enhances or corrects existing memory
3. DELETE - New information that contradicts existing memory
4. NONE - Information already captured or irrelevant

Decision logic:

ADD when:
- Fact introduces completely new information
- No existing memory covers this topic
- Assign a new unique ID

UPDATE when:
- New fact provides more detail than existing memory
- New fact corrects or refines existing information
- Keep the original ID, include "old_memory" field
- Example: "Likes pizza" → "Prefers pepperoni pizza" (UPDATE)
- Counter-example: "Likes pizza" → "Enjoys pizza" (NONE - same meaning)

DELETE when:
- New fact directly contradicts existing memory
- Example: "Vegetarian" contradicts "Loves steak"

NONE when:
- Fact already exists in memory
- Fact conveys same information with different wording
- Fact is irrelevant or generic

Response format:
{{
    "memory": [
        {{
            "id": "memory_id",
            "text": "memory content",
            "event": "ADD|UPDATE|DELETE|NONE",
            "old_memory": "previous content (only for UPDATE)"
        }}
    ]
}}
"""

def get_update_memory_messages(existing_memories: str, new_facts: str) -> str:
    """Generate the update memory prompt with context."""

    if existing_memories:
        memory_context = f"""
Existing memory state:
```json
{existing_memories}
```
"""
    else:
        memory_context = "Memory is currently empty. All facts will be new additions."

    return f"""{UPDATE_MEMORY_PROMPT}

{memory_context}

Newly extracted facts:
```json
{new_facts}
```

Instructions:
- Reconcile the new facts with existing memory
- Determine the appropriate operation (ADD/UPDATE/DELETE/NONE) for each memory entry
- Maintain consistency and avoid redundancy
- Output valid JSON only, no additional text

Required output structure:
{{
    "memory": [
        {{"id": "...", "text": "...", "event": "...", "old_memory": "..."}}
    ]
}}
"""
