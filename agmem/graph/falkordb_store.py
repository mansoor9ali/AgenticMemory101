"""
FalkorDB implementation of GraphStoreBase.
"""

import json
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    import redis.asyncio as redis_asyncio
except ImportError:  # pragma: no cover - optional dependency
    redis_asyncio = None

from agmem.graph.models import Entity, Episode, Relationship
from agmem.graph.store import GraphStoreBase

logger = logging.getLogger(__name__)


class FalkorDBGraphStore(GraphStoreBase):
    """Graph store backed by FalkorDB/RedisGraph."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._client: Optional[Any] = None
        self._graph_name = self.config.get("graph_name", "agentic_memory")
        self._initialized = False

    async def _ensure_client(self) -> None:
        if self._client is not None:
            return

        if redis_asyncio is None:
            raise ImportError(
                "redis package is required for FalkorDB support. "
                "Install with: pip install redis"
            )

        host = self.config.get("host", "localhost")
        port = int(self.config.get("port", 6379))
        user = self.config.get("user") or None
        password = self.config.get("password") or None
        use_tls = bool(self.config.get("use_tls", False))
        timeout = float(self.config.get("connection_timeout", 30.0))

        self._client = redis_asyncio.Redis(
            host=host,
            port=port,
            username=user,
            password=password,
            ssl=use_tls,
            socket_connect_timeout=timeout,
            decode_responses=True,
        )

    async def initialize(self) -> None:
        if self._initialized:
            return

        await self._ensure_client()

        try:
            await self._client.execute_command(
                "GRAPH.CONFIG", "SET", "RESULTSET_FORMAT", "COMPACT"
            )
        except Exception as exc:  # pragma: no cover - configuration optional
            logger.debug(f"Unable to set FalkorDB result format: {exc}")

        # Touch the graph so it exists before first write
        try:
            await self._client.execute_command(
                "GRAPH.QUERY",
                self._graph_name,
                "RETURN 1",
                "--compact",
            )
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Graph warm-up query failed: {exc}")

        self._initialized = True

    async def close(self) -> None:
        if self._client:
            try:
                await self._client.close()
            finally:
                await self._client.connection_pool.disconnect()  # type: ignore[attr-defined]
            self._client = None

    # ---------------------------------------------------------------------
    # Entity operations
    # ---------------------------------------------------------------------

    async def save_entity(self, entity: Entity) -> None:
        assignments = self._build_set_clause(
            "e",
            {
                "user_id": entity.user_id,
                "name": entity.name,
                "name_hash": entity.name_hash,
                "entity_type": entity.entity_type,
                "summary": entity.summary,
                "embedding": entity.embedding or [],
                "metadata": json.dumps(entity.metadata or {}),
                "created_at": entity.created_at.isoformat(),
                "updated_at": entity.updated_at.isoformat(),
            },
        )
        query = f"""
        MERGE (e:Entity {{id: {self._quote(entity.id)}}})
        SET {assignments}
        """
        await self._execute_query(query)

    async def save_entities(self, entities: List[Entity]) -> None:
        for entity in entities:
            await self.save_entity(entity)

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        query = f"""
        MATCH (e:Entity {{id: {self._quote(entity_id)}}})
        RETURN properties(e) as props
        LIMIT 1
        """
        records = await self._execute_query(query, read_only=True)
        if not records:
            return None
        return self._record_to_entity(records[0].get("props", {}))

    async def get_entities_by_user(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[Entity]:
        query = f"""
        MATCH (e:Entity {{user_id: {self._quote(user_id)}}})
        RETURN properties(e) as props
        ORDER BY e.created_at DESC
        LIMIT {limit}
        """
        records = await self._execute_query(query, read_only=True)
        return [self._record_to_entity(r.get("props", {})) for r in records]

    async def find_entity_by_name(
        self,
        name: str,
        user_id: str,
    ) -> Optional[Entity]:
        from agmem.graph.models import generate_hash

        name_hash = generate_hash(name)
        query = f"""
        MATCH (e:Entity {{
            user_id: {self._quote(user_id)},
            name_hash: {self._quote(name_hash)}
        }})
        RETURN properties(e) as props
        LIMIT 1
        """
        records = await self._execute_query(query, read_only=True)
        if not records:
            return None
        return self._record_to_entity(records[0].get("props", {}))

    async def delete_entity(self, entity_id: str) -> None:
        query = f"""
        MATCH (e:Entity {{id: {self._quote(entity_id)}}})
        DETACH DELETE e
        """
        await self._execute_query(query)

    # ---------------------------------------------------------------------
    # Relationship operations
    # ---------------------------------------------------------------------

    async def save_relationship(self, rel: Relationship) -> None:
        assignments = self._build_set_clause(
            "r",
            {
                "user_id": rel.user_id,
                "relation_type": rel.relation_type,
                "fact": rel.fact,
                "fact_hash": rel.fact_hash,
                "embedding": rel.embedding or [],
                "metadata": json.dumps(rel.metadata or {}),
                "created_at": rel.created_at.isoformat(),
                "updated_at": rel.updated_at.isoformat(),
            },
        )
        query = f"""
        MATCH (source:Entity {{id: {self._quote(rel.source_id)}}})
        MATCH (target:Entity {{id: {self._quote(rel.target_id)}}})
        MERGE (source)-[r:RELATES_TO {{id: {self._quote(rel.id)}}}]->(target)
        SET {assignments}
        """
        await self._execute_query(query)

    async def save_relationships(self, rels: List[Relationship]) -> None:
        for rel in rels:
            await self.save_relationship(rel)

    async def get_relationship(self, rel_id: str) -> Optional[Relationship]:
        query = f"""
        MATCH (source:Entity)-[r:RELATES_TO {{id: {self._quote(rel_id)}}}]->(target:Entity)
        RETURN properties(r) as rel, source.id as source_id, target.id as target_id
        LIMIT 1
        """
        records = await self._execute_query(query, read_only=True)
        if not records:
            return None
        row = records[0]
        return self._record_to_relationship(
            row.get("rel", {}),
            row.get("source_id", ""),
            row.get("target_id", ""),
        )

    async def get_entity_relationships(self, entity_id: str) -> List[Relationship]:
        query = f"""
        MATCH (e:Entity {{id: {self._quote(entity_id)}}})-[r:RELATES_TO]-(other:Entity)
        RETURN
            properties(r) as rel,
            startNode(r).id as source_id,
            endNode(r).id as target_id
        """
        records = await self._execute_query(query, read_only=True)
        return [
            self._record_to_relationship(
                row.get("rel", {}),
                row.get("source_id", ""),
                row.get("target_id", ""),
            )
            for row in records
        ]

    async def get_relationships_by_user(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[Relationship]:
        query = f"""
        MATCH (source:Entity)-[r:RELATES_TO {{
            user_id: {self._quote(user_id)}
        }}]->(target:Entity)
        RETURN properties(r) as rel, source.id as source_id, target.id as target_id
        ORDER BY r.created_at DESC
        LIMIT {limit}
        """
        records = await self._execute_query(query, read_only=True)
        return [
            self._record_to_relationship(
                row.get("rel", {}),
                row.get("source_id", ""),
                row.get("target_id", ""),
            )
            for row in records
        ]

    async def delete_relationship(self, rel_id: str) -> None:
        query = f"""
        MATCH ()-[r:RELATES_TO {{id: {self._quote(rel_id)}}}]->()
        DELETE r
        """
        await self._execute_query(query)

    # ---------------------------------------------------------------------
    # Episode operations
    # ---------------------------------------------------------------------

    async def save_episode(self, episode: Episode) -> None:
        assignments = self._build_set_clause(
            "ep",
            {
                "user_id": episode.user_id,
                "content": episode.content,
                "source": episode.source,
                "entity_ids": episode.entity_ids,
                "relationship_ids": episode.relationship_ids,
                "metadata": json.dumps(episode.metadata or {}),
                "created_at": episode.created_at.isoformat(),
            },
        )
        query = f"""
        MERGE (ep:Episode {{id: {self._quote(episode.id)}}})
        SET {assignments}
        """
        await self._execute_query(query)

    async def get_episode(self, episode_id: str) -> Optional[Episode]:
        query = f"""
        MATCH (ep:Episode {{id: {self._quote(episode_id)}}})
        RETURN properties(ep) as props
        LIMIT 1
        """
        records = await self._execute_query(query, read_only=True)
        if not records:
            return None
        return self._record_to_episode(records[0].get("props", {}))

    async def get_episodes_by_user(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[Episode]:
        query = f"""
        MATCH (ep:Episode {{user_id: {self._quote(user_id)}}})
        RETURN properties(ep) as props
        ORDER BY ep.created_at DESC
        LIMIT {limit}
        """
        records = await self._execute_query(query, read_only=True)
        return [self._record_to_episode(row.get("props", {})) for row in records]

    # ---------------------------------------------------------------------
    # Search operations
    # ---------------------------------------------------------------------

    async def search_entities(
        self,
        embedding: List[float],
        user_id: str,
        limit: int = 10,
    ) -> List[Entity]:
        entities = await self.get_entities_by_user(user_id, limit=500)
        scored: List[tuple[Entity, float]] = []
        for entity in entities:
            if entity.embedding:
                score = self._cosine_similarity(embedding, entity.embedding)
                scored.append((entity, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [entity for entity, _ in scored[:limit]]

    async def search_relationships(
        self,
        embedding: List[float],
        user_id: str,
        limit: int = 10,
    ) -> List[Relationship]:
        relationships = await self.get_relationships_by_user(user_id, limit=500)
        scored: List[tuple[Relationship, float]] = []
        for rel in relationships:
            if rel.embedding:
                score = self._cosine_similarity(embedding, rel.embedding)
                scored.append((rel, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [rel for rel, _ in scored[:limit]]

    # ---------------------------------------------------------------------
    # Graph traversal / bulk operations
    # ---------------------------------------------------------------------

    async def traverse(
        self,
        entity_id: str,
        hops: int = 2,
        user_id: Optional[str] = None,
    ) -> List[Entity]:
        user_filter = ""
        if user_id:
            user_filter = f"AND e.user_id = {self._quote(user_id)}"
        query = f"""
        MATCH (start:Entity {{id: {self._quote(entity_id)}}})
        MATCH path = (start)-[:RELATES_TO*1..{hops}]-(e:Entity)
        WHERE e.id <> start.id {user_filter}
        RETURN DISTINCT properties(e) as props
        """
        records = await self._execute_query(query, read_only=True)
        return [self._record_to_entity(row.get("props", {})) for row in records]

    async def delete_user_data(self, user_id: str) -> int:
        count_query = f"""
        MATCH (n {{user_id: {self._quote(user_id)}}})
        RETURN count(n) as cnt
        """
        records = await self._execute_query(count_query, read_only=True)
        count = 0
        if records:
            try:
                count = int(records[0].get("cnt", 0))
            except (TypeError, ValueError):
                count = 0

        delete_query = f"""
        MATCH (n {{user_id: {self._quote(user_id)}}})
        DETACH DELETE n
        """
        await self._execute_query(delete_query)
        return count

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    async def _execute_query(
        self,
        query: str,
        read_only: bool = False,
    ) -> List[Dict[str, Any]]:
        await self._ensure_client()
        command = "GRAPH.RO_QUERY" if read_only else "GRAPH.QUERY"
        compact_query = " ".join(query.split())
        response = await self._client.execute_command(
            command,
            self._graph_name,
            compact_query,
            "--compact",
        )
        return self._parse_response(response)

    def _parse_response(self, response: Any) -> List[Dict[str, Any]]:
        if not isinstance(response, (list, tuple)) or not response:
            return []

        header = response[0]
        rows = response[1] if len(response) > 1 and isinstance(response[1], (list, tuple)) else []

        if not isinstance(header, (list, tuple)) or not header:
            return []

        normalized_header = [self._normalize_header(col) for col in header]

        records: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue
            record: Dict[str, Any] = {}
            for idx, value in enumerate(row):
                if idx < len(normalized_header):
                    # Handle FalkorDB compact mode response format
                    parsed_value = self._parse_compact_value(value)
                    record[normalized_header[idx]] = parsed_value
            if record:
                records.append(record)
        return records

    def _parse_compact_value(self, value: Any) -> Any:
        """Parse a value from FalkorDB compact mode response.

        FalkorDB compact mode returns values in the format:
        - Scalars: [type, value] where type is: 1=null, 2=string, 3=int, 4=bool, 5=double
        - Maps/Properties: [10, [key1, [type, val1], key2, [type, val2], ...]]
        - Arrays: [6, [...]]
        - Nodes: [8, [...]]
        """
        if not isinstance(value, (list, tuple)):
            return value

        if len(value) == 0:
            return value

        # Check if this is a typed value [type_id, data]
        if len(value) == 2 and isinstance(value[0], int):
            type_id = value[0]
            data = value[1]

            if type_id == 1:  # NULL
                return None
            elif type_id == 2:  # STRING
                return data
            elif type_id == 3:  # INTEGER
                return int(data) if data is not None else 0
            elif type_id == 4:  # BOOLEAN
                return bool(data)
            elif type_id == 5:  # DOUBLE
                return float(data) if data is not None else 0.0
            elif type_id == 6:  # ARRAY
                if isinstance(data, (list, tuple)):
                    return [self._parse_compact_value(item) for item in data]
                return data
            elif type_id == 10:  # MAP (properties)
                return self._parse_compact_map(data)
            elif type_id == 8:  # NODE - extract properties
                # Node format: [8, [node_id, [labels], [properties]]]
                if isinstance(data, (list, tuple)) and len(data) >= 3:
                    return self._parse_compact_value(data[2])
                return {}

        # Check if this looks like a nested compact value
        if len(value) >= 1 and isinstance(value[0], (list, tuple)):
            # Could be a node or relationship wrapper
            return self._parse_compact_value(value[0])

        # Otherwise return as-is
        return value

    def _parse_compact_map(self, data: Any) -> Dict[str, Any]:
        """Parse a FalkorDB compact map/properties.

        Format: [key1, [type, val1], key2, [type, val2], ...]
        """
        if not isinstance(data, (list, tuple)):
            return {}

        result: Dict[str, Any] = {}
        i = 0
        while i < len(data) - 1:
            key = data[i]
            if isinstance(key, str):
                value = data[i + 1]
                result[key] = self._parse_compact_value(value)
                i += 2
            else:
                i += 1

        return result

    def _normalize_header(self, header: Any) -> str:
        if isinstance(header, str):
            return header
        if isinstance(header, (list, tuple)) and header:
            return str(header[-1])
        return ""

    def _build_set_clause(self, alias: str, props: Dict[str, Any]) -> str:
        assignments = []
        for key, value in props.items():
            assignments.append(f"{alias}.{key} = {self._format_value(value)}")
        return ", ".join(assignments)

    def _format_value(self, value: Any) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, (list, tuple)):
            inner = ", ".join(self._format_value(item) for item in value)
            return f"[{inner}]"
        if isinstance(value, dict):
            return self._quote(json.dumps(value))
        return self._quote(value)

    def _quote(self, value: Any) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        text = str(value)
        text = text.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{text}'"

    def _record_to_entity(self, props: Dict[str, Any]) -> Entity:
        metadata = self._decode_metadata(props.get("metadata"))
        embedding = self._decode_embedding(props.get("embedding"))
        created_at = self._parse_datetime(props.get("created_at"))
        updated_at = self._parse_datetime(props.get("updated_at"))

        return Entity(
            id=props.get("id", ""),
            user_id=props.get("user_id", ""),
            name=props.get("name", ""),
            entity_type=props.get("entity_type", "entity"),
            summary=props.get("summary", ""),
            metadata=metadata,
            embedding=embedding,
            created_at=created_at,
            updated_at=updated_at,
            name_hash=props.get("name_hash", ""),
        )

    def _record_to_relationship(
        self,
        props: Dict[str, Any],
        source_id: str,
        target_id: str,
    ) -> Relationship:
        metadata = self._decode_metadata(props.get("metadata"))
        embedding = self._decode_embedding(props.get("embedding"))
        created_at = self._parse_datetime(props.get("created_at"))
        updated_at = self._parse_datetime(props.get("updated_at"))

        return Relationship(
            id=props.get("id", ""),
            user_id=props.get("user_id", ""),
            source_id=source_id,
            target_id=target_id,
            relation_type=props.get("relation_type", ""),
            fact=props.get("fact", ""),
            metadata=metadata,
            embedding=embedding,
            created_at=created_at,
            updated_at=updated_at,
            fact_hash=props.get("fact_hash", ""),
        )

    def _record_to_episode(self, props: Dict[str, Any]) -> Episode:
        metadata = self._decode_metadata(props.get("metadata"))
        entity_ids = self._decode_list(props.get("entity_ids"))
        relationship_ids = self._decode_list(props.get("relationship_ids"))
        created_at = self._parse_datetime(props.get("created_at"))

        return Episode(
            id=props.get("id", ""),
            user_id=props.get("user_id", ""),
            content=props.get("content", ""),
            source=props.get("source", "message"),
            entity_ids=entity_ids,
            relationship_ids=relationship_ids,
            metadata=metadata,
            created_at=created_at,
        )

    def _decode_metadata(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value:
            try:
                data = json.loads(value)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        return {}

    def _decode_list(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str) and value:
            try:
                data = json.loads(value)
                if isinstance(data, list):
                    return [str(item) for item in data]
            except json.JSONDecodeError:
                pass
        return []

    def _decode_embedding(self, value: Any) -> Optional[List[float]]:
        data = None
        if isinstance(value, list):
            data = value
        elif isinstance(value, str) and value:
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    data = parsed
            except json.JSONDecodeError:
                data = None

        if data is None:
            return None

        floats: List[float] = []
        for item in data:
            try:
                floats.append(float(item))
            except (TypeError, ValueError):
                continue
        return floats or None

    def _parse_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass
        return datetime.utcnow()

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
