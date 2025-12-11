"""
Data models for Graph Memory.
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


def generate_hash(content: str) -> str:
    """Generate hash for content deduplication."""
    return hashlib.md5(content.lower().strip().encode()).hexdigest()


@dataclass
class Entity:
    """
    A node in the knowledge graph.
    
    Represents a person, place, thing, concept, or any named entity
    extracted from conversations.
    """
    
    id: str
    user_id: str
    name: str
    entity_type: str = "entity"  # person, company, technology, place, etc.
    summary: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # For deduplication
    name_hash: str = ""
    
    def __post_init__(self):
        if not self.name_hash:
            self.name_hash = generate_hash(self.name)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "summary": self.summary,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        data = data.copy()
        for field_name in ["created_at", "updated_at"]:
            if field_name in data and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])
        # Remove embedding from dict if present (handled separately)
        data.pop("embedding", None)
        return cls(**data)


@dataclass
class Relationship:
    """
    An edge in the knowledge graph.
    
    Represents a fact connecting two entities, e.g., "John works at Acme".
    """
    
    id: str
    user_id: str
    source_id: str  # Entity ID
    target_id: str  # Entity ID
    relation_type: str  # works_at, likes, located_in, knows, etc.
    fact: str  # Human-readable: "John works at Acme Corp"
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # For deduplication
    fact_hash: str = ""
    
    def __post_init__(self):
        if not self.fact_hash:
            self.fact_hash = generate_hash(self.fact)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "fact": self.fact,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        data = data.copy()
        for field_name in ["created_at", "updated_at"]:
            if field_name in data and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])
        data.pop("embedding", None)
        return cls(**data)


@dataclass
class Episode:
    """
    A conversation or text that was processed.
    
    Tracks which entities and relationships were extracted from a piece of content.
    """
    
    id: str
    user_id: str
    content: str
    source: str = "message"  # message, text, document
    entity_ids: List[str] = field(default_factory=list)
    relationship_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "source": self.source,
            "entity_ids": self.entity_ids,
            "relationship_ids": self.relationship_ids,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        data = data.copy()
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class GraphSearchResult:
    """Result from graph search."""
    
    id: str
    fact: str
    relation_type: str
    source_entity: str
    target_entity: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "fact": self.fact,
            "relation_type": self.relation_type,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "score": self.score,
            "metadata": self.metadata,
        }
