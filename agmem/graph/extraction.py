"""
Entity and relationship extraction from text.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from agmem.graph.models import Entity, Relationship, generate_id
from agmem.graph.prompts import (
    ENTITY_EXTRACTION_PROMPT,
    RELATIONSHIP_EXTRACTION_PROMPT,
)

logger = logging.getLogger(__name__)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in markdown code block
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"\{[\s\S]*\}",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                json_str = match.group(1) if "```" in pattern else match.group(0)
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                continue
    
    return None


async def extract_entities(
    llm,
    content: str,
    max_entities: int = 10,
) -> List[Dict[str, Any]]:
    """
    Extract entities from content using LLM.
    
    Returns list of dicts with: name, type, summary
    """
    prompt = ENTITY_EXTRACTION_PROMPT.format(
        content=content,
        max_entities=max_entities,
    )
    
    response = llm.generate_response(
        messages=[{"role": "user", "content": prompt}]
    )
    
    result = extract_json(response.get("content", ""))
    
    if result and "entities" in result:
        return result["entities"]
    
    return []


async def extract_relationships(
    llm,
    content: str,
    entities: List[Dict[str, Any]],
    max_relationships: int = 15,
) -> List[Dict[str, Any]]:
    """
    Extract relationships between entities using LLM.
    
    Returns list of dicts with: source, target, relation_type, fact
    """
    if not entities:
        return []
    
    # Format entities for prompt
    entity_list = "\n".join([
        f"- {e['name']} ({e['type']}): {e.get('summary', '')}"
        for e in entities
    ])
    
    prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
        content=content,
        entities=entity_list,
        max_relationships=max_relationships,
    )
    
    response = llm.generate_response(
        messages=[{"role": "user", "content": prompt}]
    )
    
    result = extract_json(response.get("content", ""))
    
    if result and "relationships" in result:
        return result["relationships"]
    
    return []


def normalize_entity_name(name: str) -> str:
    """Normalize entity name for comparison."""
    return name.lower().strip()


def find_matching_entity(
    name: str,
    existing_entities: List[Entity],
    threshold: float = 0.9,
) -> Optional[Entity]:
    """
    Find existing entity that matches the name.
    
    Uses simple string matching. For production, consider
    using embeddings or LLM-based deduplication.
    """
    normalized = normalize_entity_name(name)
    
    for entity in existing_entities:
        existing_normalized = normalize_entity_name(entity.name)
        
        # Exact match
        if normalized == existing_normalized:
            return entity
        
        # One contains the other (e.g., "John" in "John Smith")
        if normalized in existing_normalized or existing_normalized in normalized:
            # Only match if significant overlap
            shorter = min(len(normalized), len(existing_normalized))
            longer = max(len(normalized), len(existing_normalized))
            if shorter / longer >= threshold:
                return entity
    
    return None


def create_entities_from_extraction(
    extracted: List[Dict[str, Any]],
    user_id: str,
    existing_entities: List[Entity],
    dedup_threshold: float = 0.9,
) -> Tuple[List[Entity], Dict[str, str]]:
    """
    Create Entity objects from extracted data, deduplicating against existing.
    
    Returns:
        - List of new entities to save
        - Dict mapping extracted names to entity IDs (for relationship creation)
    """
    new_entities = []
    name_to_id = {}
    
    # First, map existing entities
    for entity in existing_entities:
        name_to_id[normalize_entity_name(entity.name)] = entity.id
    
    for item in extracted:
        name = item.get("name", "").strip()
        if not name:
            continue
        
        normalized = normalize_entity_name(name)
        
        # Check if already mapped
        if normalized in name_to_id:
            continue
        
        # Check against existing entities
        match = find_matching_entity(name, existing_entities, dedup_threshold)
        if match:
            name_to_id[normalized] = match.id
            continue
        
        # Check against new entities we're creating
        match = find_matching_entity(name, new_entities, dedup_threshold)
        if match:
            name_to_id[normalized] = match.id
            continue
        
        # Create new entity
        entity = Entity(
            id=generate_id(),
            user_id=user_id,
            name=name,
            entity_type=item.get("type", "entity"),
            summary=item.get("summary", ""),
        )
        
        new_entities.append(entity)
        name_to_id[normalized] = entity.id
    
    return new_entities, name_to_id


def create_relationships_from_extraction(
    extracted: List[Dict[str, Any]],
    user_id: str,
    name_to_id: Dict[str, str],
    existing_relationships: List[Relationship],
) -> List[Relationship]:
    """
    Create Relationship objects from extracted data.
    
    Deduplicates against existing relationships.
    """
    new_relationships = []
    existing_facts = {r.fact_hash for r in existing_relationships}
    
    for item in extracted:
        source_name = normalize_entity_name(item.get("source", ""))
        target_name = normalize_entity_name(item.get("target", ""))
        
        # Get entity IDs
        source_id = name_to_id.get(source_name)
        target_id = name_to_id.get(target_name)
        
        if not source_id or not target_id:
            logger.debug(f"Skipping relationship: missing entity for {source_name} -> {target_name}")
            continue
        
        fact = item.get("fact", "")
        if not fact:
            continue
        
        # Check for duplicate
        from agmem.graph.models import generate_hash
        fact_hash = generate_hash(fact)
        
        if fact_hash in existing_facts:
            continue
        
        # Check if we already created this relationship
        if fact_hash in {r.fact_hash for r in new_relationships}:
            continue
        
        relationship = Relationship(
            id=generate_id(),
            user_id=user_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=item.get("relation_type", "related_to"),
            fact=fact,
        )
        
        new_relationships.append(relationship)
        existing_facts.add(fact_hash)
    
    return new_relationships


async def extract_graph_from_content(
    llm,
    embedder,
    content: str,
    user_id: str,
    existing_entities: List[Entity],
    existing_relationships: List[Relationship],
    max_entities: int = 10,
    max_relationships: int = 15,
    dedup_threshold: float = 0.9,
) -> Tuple[List[Entity], List[Relationship]]:
    """
    Full extraction pipeline: content -> entities + relationships.
    
    Returns new entities and relationships to save.
    """
    # Extract entities
    extracted_entities = await extract_entities(llm, content, max_entities)
    
    if not extracted_entities:
        return [], []
    
    # Create entity objects with deduplication
    new_entities, name_to_id = create_entities_from_extraction(
        extracted_entities,
        user_id,
        existing_entities,
        dedup_threshold,
    )
    
    # Generate embeddings for new entities
    for entity in new_entities:
        entity.embedding = embedder.embed(entity.name)
    
    # Extract relationships
    extracted_rels = await extract_relationships(
        llm,
        content,
        extracted_entities,
        max_relationships,
    )
    
    # Create relationship objects
    new_relationships = create_relationships_from_extraction(
        extracted_rels,
        user_id,
        name_to_id,
        existing_relationships,
    )
    
    # Generate embeddings for new relationships
    for rel in new_relationships:
        rel.embedding = embedder.embed(rel.fact)
    
    return new_entities, new_relationships
