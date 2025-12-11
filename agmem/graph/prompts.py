"""
Prompts for entity and relationship extraction.
"""

ENTITY_EXTRACTION_PROMPT = """You are an entity extraction system. Extract named entities from the conversation.

For each entity, provide:
- name: The entity name (proper noun, normalized)
- type: One of: person, company, technology, place, product, event, concept, other
- summary: Brief description based on context (1 sentence max)

Rules:
1. Extract specific, named entities only (not generic terms like "the company")
2. Normalize names (e.g., "John" and "John Smith" referring to same person = "John Smith")
3. Include entities mentioned by any speaker
4. Maximum {max_entities} entities

Respond with JSON only:
{{
    "entities": [
        {{"name": "John Smith", "type": "person", "summary": "A software developer"}},
        {{"name": "Acme Corp", "type": "company", "summary": "John's employer"}}
    ]
}}

Conversation:
{content}"""


RELATIONSHIP_EXTRACTION_PROMPT = """You are a relationship extraction system. Given entities and conversation, extract relationships between them.

Known entities:
{entities}

For each relationship, provide:
- source: Source entity name (must be from the list above)
- target: Target entity name (must be from the list above)
- relation_type: Relationship type (lowercase, underscore-separated)
- fact: Human-readable fact sentence

Common relation types:
- works_at, employed_by
- lives_in, located_in
- likes, dislikes, prefers
- knows, friends_with
- uses, built_with
- owns, created_by
- part_of, member_of

Rules:
1. Only use entities from the provided list
2. Each relationship should be a clear fact from the conversation
3. Avoid duplicate or redundant relationships
4. Maximum {max_relationships} relationships

Respond with JSON only:
{{
    "relationships": [
        {{
            "source": "John Smith",
            "target": "Acme Corp",
            "relation_type": "works_at",
            "fact": "John Smith works at Acme Corp"
        }},
        {{
            "source": "John Smith",
            "target": "Python",
            "relation_type": "uses",
            "fact": "John Smith uses Python for development"
        }}
    ]
}}

Conversation:
{content}"""


ENTITY_DEDUP_PROMPT = """You are an entity deduplication system. Determine if two entities refer to the same real-world entity.

Entity A: {entity_a}
Entity B: {entity_b}

Consider:
- Name variations (John vs John Smith)
- Abbreviations (IBM vs International Business Machines)
- Context clues

Respond with JSON only:
{{
    "is_same": true/false,
    "confidence": 0.0-1.0,
    "canonical_name": "preferred name if same entity"
}}"""
