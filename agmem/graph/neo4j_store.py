"""
Neo4j implementation of GraphStoreBase.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from agmem.graph.models import Entity, Episode, Relationship
from agmem.graph.store import GraphStoreBase

logger = logging.getLogger(__name__)


class Neo4jGraphStore(GraphStoreBase):
    """Neo4j graph store implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._driver = None
        self._initialized = False
    
    async def _ensure_driver(self):
        """Lazily initialize Neo4j driver."""
        if self._driver is not None:
            return
        
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError:
            raise ImportError(
                "neo4j package required. Install with: pip install neo4j"
            )
        
        uri = self.config.get("uri", "bolt://localhost:7687")
        user = self.config.get("user", "neo4j")
        password = self.config.get("password", "")
        max_pool = self.config.get("max_connection_pool_size", 50)
        
        self._driver = AsyncGraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=max_pool,
        )
        self._database = self.config.get("database", "neo4j")
    
    async def initialize(self) -> None:
        """Create indices for efficient queries."""
        if self._initialized:
            return
        
        await self._ensure_driver()
        
        indices = [
            # Entity indices
            "CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_user IF NOT EXISTS FOR (e:Entity) ON (e.user_id)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_hash IF NOT EXISTS FOR (e:Entity) ON (e.name_hash)",
            # Relationship indices (on relationship properties)
            "CREATE INDEX rel_id IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.id)",
            "CREATE INDEX rel_user IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.user_id)",
            # Episode indices
            "CREATE INDEX episode_id IF NOT EXISTS FOR (ep:Episode) ON (ep.id)",
            "CREATE INDEX episode_user IF NOT EXISTS FOR (ep:Episode) ON (ep.user_id)",
        ]
        
        async with self._driver.session(database=self._database) as session:
            for idx in indices:
                try:
                    await session.run(idx)
                except Exception as e:
                    logger.debug(f"Index creation note: {e}")
        
        self._initialized = True
        logger.info("Neo4j graph store initialized")
    
    # Entity operations
    
    async def save_entity(self, entity: Entity) -> None:
        """Save an entity node."""
        await self._ensure_driver()
        
        query = """
        MERGE (e:Entity {id: $id})
        SET e.user_id = $user_id,
            e.name = $name,
            e.name_hash = $name_hash,
            e.entity_type = $entity_type,
            e.summary = $summary,
            e.embedding = $embedding,
            e.metadata = $metadata,
            e.created_at = $created_at,
            e.updated_at = $updated_at
        """
        
        async with self._driver.session(database=self._database) as session:
            await session.run(
                query,
                id=entity.id,
                user_id=entity.user_id,
                name=entity.name,
                name_hash=entity.name_hash,
                entity_type=entity.entity_type,
                summary=entity.summary,
                embedding=entity.embedding,
                metadata=json.dumps(entity.metadata) if entity.metadata else "{}",
                created_at=entity.created_at.isoformat(),
                updated_at=entity.updated_at.isoformat(),
            )
    
    async def save_entities(self, entities: List[Entity]) -> None:
        """Save multiple entities."""
        for entity in entities:
            await self.save_entity(entity)
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        await self._ensure_driver()
        
        query = """
        MATCH (e:Entity {id: $id})
        RETURN e
        """
        
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, id=entity_id)
            record = await result.single()
            
            if record:
                return self._record_to_entity(record["e"])
            return None
    
    async def get_entities_by_user(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[Entity]:
        """Get all entities for a user."""
        await self._ensure_driver()
        
        query = """
        MATCH (e:Entity {user_id: $user_id})
        RETURN e
        ORDER BY e.created_at DESC
        LIMIT $limit
        """
        
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, user_id=user_id, limit=limit)
            records = await result.data()
            return [self._record_to_entity(r["e"]) for r in records]
    
    async def find_entity_by_name(
        self,
        name: str,
        user_id: str,
    ) -> Optional[Entity]:
        """Find entity by name (case-insensitive)."""
        await self._ensure_driver()
        
        from agmem.graph.models import generate_hash
        name_hash = generate_hash(name)
        
        query = """
        MATCH (e:Entity {user_id: $user_id, name_hash: $name_hash})
        RETURN e
        LIMIT 1
        """
        
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, user_id=user_id, name_hash=name_hash)
            record = await result.single()
            
            if record:
                return self._record_to_entity(record["e"])
            return None
    
    async def delete_entity(self, entity_id: str) -> None:
        """Delete entity and its relationships."""
        await self._ensure_driver()
        
        query = """
        MATCH (e:Entity {id: $id})
        DETACH DELETE e
        """
        
        async with self._driver.session(database=self._database) as session:
            await session.run(query, id=entity_id)
    
    # Relationship operations
    
    async def save_relationship(self, rel: Relationship) -> None:
        """Save a relationship edge."""
        await self._ensure_driver()
        
        query = """
        MATCH (source:Entity {id: $source_id})
        MATCH (target:Entity {id: $target_id})
        MERGE (source)-[r:RELATES_TO {id: $id}]->(target)
        SET r.user_id = $user_id,
            r.relation_type = $relation_type,
            r.fact = $fact,
            r.fact_hash = $fact_hash,
            r.embedding = $embedding,
            r.metadata = $metadata,
            r.created_at = $created_at,
            r.updated_at = $updated_at
        """
        
        async with self._driver.session(database=self._database) as session:
            await session.run(
                query,
                id=rel.id,
                source_id=rel.source_id,
                target_id=rel.target_id,
                user_id=rel.user_id,
                relation_type=rel.relation_type,
                fact=rel.fact,
                fact_hash=rel.fact_hash,
                embedding=rel.embedding,
                metadata=json.dumps(rel.metadata) if rel.metadata else "{}",
                created_at=rel.created_at.isoformat(),
                updated_at=rel.updated_at.isoformat(),
            )
    
    async def save_relationships(self, rels: List[Relationship]) -> None:
        """Save multiple relationships."""
        for rel in rels:
            await self.save_relationship(rel)
    
    async def get_relationship(self, rel_id: str) -> Optional[Relationship]:
        """Get relationship by ID."""
        await self._ensure_driver()
        
        query = """
        MATCH (source:Entity)-[r:RELATES_TO {id: $id}]->(target:Entity)
        RETURN r, source.id AS source_id, target.id AS target_id
        """
        
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, id=rel_id)
            record = await result.single()
            
            if record:
                return self._record_to_relationship(
                    record["r"],
                    record["source_id"],
                    record["target_id"],
                )
            return None
    
    async def get_entity_relationships(
        self,
        entity_id: str,
    ) -> List[Relationship]:
        """Get all relationships for an entity."""
        await self._ensure_driver()
        
        query = """
        MATCH (e:Entity {id: $entity_id})-[r:RELATES_TO]-(other:Entity)
        WITH r, 
             CASE WHEN startNode(r).id = $entity_id THEN startNode(r).id ELSE endNode(r).id END AS source_id,
             CASE WHEN startNode(r).id = $entity_id THEN endNode(r).id ELSE startNode(r).id END AS target_id
        RETURN DISTINCT r, source_id, target_id
        """
        
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, entity_id=entity_id)
            records = await result.data()
            return [
                self._record_to_relationship(r["r"], r["source_id"], r["target_id"])
                for r in records
            ]
    
    async def get_relationships_by_user(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[Relationship]:
        """Get all relationships for a user."""
        await self._ensure_driver()
        
        query = """
        MATCH (source:Entity)-[r:RELATES_TO {user_id: $user_id}]->(target:Entity)
        RETURN r, source.id AS source_id, target.id AS target_id
        ORDER BY r.created_at DESC
        LIMIT $limit
        """
        
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, user_id=user_id, limit=limit)
            records = await result.data()
            return [
                self._record_to_relationship(r["r"], r["source_id"], r["target_id"])
                for r in records
            ]
    
    async def delete_relationship(self, rel_id: str) -> None:
        """Delete a relationship."""
        await self._ensure_driver()
        
        query = """
        MATCH ()-[r:RELATES_TO {id: $id}]->()
        DELETE r
        """
        
        async with self._driver.session(database=self._database) as session:
            await session.run(query, id=rel_id)
    
    # Episode operations
    
    async def save_episode(self, episode: Episode) -> None:
        """Save an episode."""
        await self._ensure_driver()
        
        query = """
        MERGE (ep:Episode {id: $id})
        SET ep.user_id = $user_id,
            ep.content = $content,
            ep.source = $source,
            ep.entity_ids = $entity_ids,
            ep.relationship_ids = $relationship_ids,
            ep.metadata = $metadata,
            ep.created_at = $created_at
        """
        
        async with self._driver.session(database=self._database) as session:
            await session.run(
                query,
                id=episode.id,
                user_id=episode.user_id,
                content=episode.content,
                source=episode.source,
                entity_ids=episode.entity_ids,
                relationship_ids=episode.relationship_ids,
                metadata=json.dumps(episode.metadata) if episode.metadata else "{}",
                created_at=episode.created_at.isoformat(),
            )
    
    async def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get episode by ID."""
        await self._ensure_driver()
        
        query = """
        MATCH (ep:Episode {id: $id})
        RETURN ep
        """
        
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, id=episode_id)
            record = await result.single()
            
            if record:
                return self._record_to_episode(record["ep"])
            return None
    
    async def get_episodes_by_user(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[Episode]:
        """Get episodes for a user."""
        await self._ensure_driver()
        
        query = """
        MATCH (ep:Episode {user_id: $user_id})
        RETURN ep
        ORDER BY ep.created_at DESC
        LIMIT $limit
        """
        
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, user_id=user_id, limit=limit)
            records = await result.data()
            return [self._record_to_episode(r["ep"]) for r in records]
    
    # Search operations
    
    async def search_entities(
        self,
        embedding: List[float],
        user_id: str,
        limit: int = 10,
    ) -> List[Entity]:
        """Vector search on entities."""
        await self._ensure_driver()
        
        # Get all entities and compute similarity in Python
        # (Neo4j Community doesn't have native vector search)
        entities = await self.get_entities_by_user(user_id, limit=500)
        
        if not entities:
            return []
        
        # Filter entities with embeddings and compute scores
        scored = []
        for entity in entities:
            if entity.embedding:
                score = self._cosine_similarity(embedding, entity.embedding)
                scored.append((entity, score))
        
        # Sort by score and return top results
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:limit]]
    
    async def search_relationships(
        self,
        embedding: List[float],
        user_id: str,
        limit: int = 10,
    ) -> List[Relationship]:
        """Vector search on relationships."""
        await self._ensure_driver()
        
        rels = await self.get_relationships_by_user(user_id, limit=500)
        
        if not rels:
            return []
        
        scored = []
        for rel in rels:
            if rel.embedding:
                score = self._cosine_similarity(embedding, rel.embedding)
                scored.append((rel, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored[:limit]]
    
    # Graph traversal
    
    async def traverse(
        self,
        entity_id: str,
        hops: int = 2,
        user_id: Optional[str] = None,
    ) -> List[Entity]:
        """BFS traversal from entity."""
        await self._ensure_driver()
        
        # Build query with variable path length
        user_filter = "AND e.user_id = $user_id" if user_id else ""
        
        query = f"""
        MATCH (start:Entity {{id: $entity_id}})
        MATCH path = (start)-[:RELATES_TO*1..{hops}]-(e:Entity)
        WHERE e.id <> $entity_id {user_filter}
        RETURN DISTINCT e
        """
        
        params = {"entity_id": entity_id}
        if user_id:
            params["user_id"] = user_id
        
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, **params)
            records = await result.data()
            return [self._record_to_entity(r["e"]) for r in records]
    
    # Bulk operations
    
    async def delete_user_data(self, user_id: str) -> int:
        """Delete all graph data for a user."""
        await self._ensure_driver()
        
        # Count before delete
        count_query = """
        MATCH (n {user_id: $user_id})
        RETURN count(n) AS count
        """
        
        delete_query = """
        MATCH (n {user_id: $user_id})
        DETACH DELETE n
        """
        
        async with self._driver.session(database=self._database) as session:
            result = await session.run(count_query, user_id=user_id)
            record = await result.single()
            count = record["count"] if record else 0
            
            await session.run(delete_query, user_id=user_id)
            return count
    
    async def close(self) -> None:
        """Close the driver."""
        if self._driver:
            await self._driver.close()
            self._driver = None
    
    # Helper methods
    
    def _record_to_entity(self, node) -> Entity:
        """Convert Neo4j node to Entity."""
        from datetime import datetime
        
        # Handle both dict (from result.data()) and Neo4j node object
        if node is None or node == "":
            props = {}
        elif isinstance(node, dict):
            props = node
        elif hasattr(node, "_properties"):
            props = dict(node._properties)
        else:
            try:
                props = dict(node)
            except (ValueError, TypeError):
                props = {}
        
        # Parse dates
        created_at = props.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        updated_at = props.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        return Entity(
            id=props.get("id", ""),
            user_id=props.get("user_id", ""),
            name=props.get("name", ""),
            name_hash=props.get("name_hash", ""),
            entity_type=props.get("entity_type", "entity"),
            summary=props.get("summary", ""),
            embedding=props.get("embedding"),
            metadata=json.loads(props.get("metadata", "{}") or "{}"),
            created_at=created_at or datetime.utcnow(),
            updated_at=updated_at or datetime.utcnow(),
        )
    
    def _record_to_relationship(
        self,
        rel,
        source_id: str,
        target_id: str,
    ) -> Relationship:
        """Convert Neo4j relationship to Relationship."""
        from datetime import datetime
        
        # Handle both dict (from result.data()) and Neo4j relationship object
        if rel is None or rel == "":
            props = {}
        elif isinstance(rel, dict):
            props = rel
        elif hasattr(rel, "_properties"):
            # Neo4j relationship object
            props = dict(rel._properties)
        else:
            try:
                props = dict(rel)
            except (ValueError, TypeError):
                props = {}
        
        created_at = props.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        updated_at = props.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        return Relationship(
            id=props.get("id", ""),
            user_id=props.get("user_id", ""),
            source_id=source_id,
            target_id=target_id,
            relation_type=props.get("relation_type", ""),
            fact=props.get("fact", ""),
            fact_hash=props.get("fact_hash", ""),
            embedding=props.get("embedding"),
            metadata=json.loads(props.get("metadata", "{}") or "{}"),
            created_at=created_at or datetime.utcnow(),
            updated_at=updated_at or datetime.utcnow(),
        )
    
    def _record_to_episode(self, node) -> Episode:
        """Convert Neo4j node to Episode."""
        from datetime import datetime
        
        # Handle both dict (from result.data()) and Neo4j node object
        if node is None or node == "":
            props = {}
        elif isinstance(node, dict):
            props = node
        elif hasattr(node, "_properties"):
            props = dict(node._properties)
        else:
            try:
                props = dict(node)
            except (ValueError, TypeError):
                props = {}
        
        created_at = props.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return Episode(
            id=props.get("id", ""),
            user_id=props.get("user_id", ""),
            content=props.get("content", ""),
            source=props.get("source", "message"),
            entity_ids=props.get("entity_ids", []),
            relationship_ids=props.get("relationship_ids", []),
            metadata=json.loads(props.get("metadata", "{}") or "{}"),
            created_at=created_at or datetime.utcnow(),
        )
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)
