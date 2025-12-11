"""
Core data models for AgenticMemory.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid
import hashlib


def generate_id() -> str:
    """Generate a unique memory ID."""
    return str(uuid.uuid4())


def generate_hash(content: str) -> str:
    """Generate a hash for memory content."""
    return hashlib.md5(content.encode()).hexdigest()


@dataclass
class Memory:
    """Represents a single memory unit."""
    
    id: str
    user_id: str
    content: str
    hash: str = ""
    embedding: Optional[List[float]] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    source: str = "manual"  # manual, conversation, extraction
    
    # Scoring factors
    importance: float = 0.5
    decay_rate: float = 0.01
    access_count: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed_at: datetime = field(default_factory=datetime.utcnow)
    
    # Graph
    links: List[str] = field(default_factory=list)
    
    # Consolidation state
    is_consolidated: bool = False
    consolidated_from: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.hash:
            self.hash = generate_hash(self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "hash": self.hash,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "tags": self.tags,
            "source": self.source,
            "importance": self.importance,
            "decay_rate": self.decay_rate,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_accessed_at": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            "links": self.links,
            "is_consolidated": self.is_consolidated,
            "consolidated_from": self.consolidated_from,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        data = data.copy()
        for field_name in ["created_at", "updated_at", "last_accessed_at"]:
            if field_name in data and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])
        return cls(**data)


@dataclass
class MemoryInput:
    """Input for creating a new memory."""
    
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5
    source: str = "manual"


@dataclass
class SearchResult:
    """Result from memory search."""
    
    id: str
    memory: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "memory": self.memory,
            "score": self.score,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
