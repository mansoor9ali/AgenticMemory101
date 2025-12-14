"""
AgenticMemory Test Script with Multi-Tenancy Support.

This script demonstrates:
- Vector memory with user isolation
- Graph memory with tenant isolation
- Multi-tenant memory (userId, agentId, sessionId)
"""

from agmem import AsyncMemory, AsyncGraphMemory, MultiTenantMemory, MultiTenantGraphMemory
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_base_config() -> dict:
    """Get the base configuration from environment variables."""
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": os.getenv("LLM_MODEL"),
                "base_url": os.getenv("LLM_BASE_URL"),
                "api_key": os.getenv("LLM_API_KEY"),
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": os.getenv("EMBEDDER_MODEL"),
                "base_url": os.getenv("EMBEDDER_BASE_URL"),
                "api_key": os.getenv("EMBEDDER_API_KEY"),
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "url": os.getenv("QDRANT_URL", "http://localhost:6333"),
            },
        },
        "storage": {
            "provider": "postgres",
            "config": {
                "host": os.getenv("POSTGRES_HOST", "localhost"),
                "port": int(os.getenv("POSTGRES_PORT", "5432")),
                "database": os.getenv("POSTGRES_DATABASE", "agenticMermoryDB"),
                "user": os.getenv("POSTGRES_USER", "user-name"),
                "password": os.getenv("POSTGRES_PASSWORD", "strong-password"),
            },
        },
        "cache": {
            "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
            "enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
        },
    }


def get_graph_config() -> dict:
    """Get graph memory configuration from environment variables."""
    return {
        "graph_store": {
            "provider": "neo4j",
            "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "user": os.getenv("NEO4J_USER", "neo4j"),
            "password": os.getenv("NEO4J_PASSWORD", "strong-password"),
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": os.getenv("LLM_MODEL"),
                "base_url": os.getenv("LLM_BASE_URL"),
                "api_key": os.getenv("LLM_API_KEY"),
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": os.getenv("EMBEDDER_MODEL"),
                "base_url": os.getenv("EMBEDDER_BASE_URL"),
                "api_key": os.getenv("EMBEDDER_API_KEY"),
            },
        },
        "max_entities_per_message": int(os.getenv("MAX_ENTITIES_PER_MESSAGE", "10")),
        "max_relationships_per_message": int(os.getenv("MAX_RELATIONSHIPS_PER_MESSAGE", "15")),
    }


def get_falkordb_config() -> dict:
    """Get FalkorDB graph memory configuration from environment variables."""
    return {
        "graph_store": {
            "provider": "falkordb",
            "host": os.getenv("FALKORDB_HOST", "localhost"),
            "port": int(os.getenv("FALKORDB_PORT", "6380")),
            "user": os.getenv("FALKORDB_USER", ""),
            "password": os.getenv("FALKORDB_PASSWORD", ""),
            "graph_name": os.getenv("FALKORDB_GRAPH_NAME", "agentic_memory"),
            "use_tls": os.getenv("FALKORDB_USE_TLS", "false").lower() == "true",
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": os.getenv("LLM_MODEL"),
                "base_url": os.getenv("LLM_BASE_URL"),
                "api_key": os.getenv("LLM_API_KEY"),
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": os.getenv("EMBEDDER_MODEL"),
                "base_url": os.getenv("EMBEDDER_BASE_URL"),
                "api_key": os.getenv("EMBEDDER_API_KEY"),
            },
        },
        "max_entities_per_message": int(os.getenv("MAX_ENTITIES_PER_MESSAGE", "10")),
        "max_relationships_per_message": int(os.getenv("MAX_RELATIONSHIPS_PER_MESSAGE", "15")),
    }


async def test_basic_memory():
    """Test basic vector memory."""
    print("\n" + "=" * 60)
    print("Testing Basic Vector Memory")
    print("=" * 60)

    config = get_base_config()
    memory = AsyncMemory(config)

    try:
        # Add memories for a user
        result = await memory.add(
            "I prefer dark mode and use Python for development",
            user_id="user_123",
        )
        print(f"Added memories: {result}")

        # Search memories
        search_result = await memory.search(
            "What are my preferences?",
            user_id="user_123",
            limit=5,
        )
        print(f"Search results: {search_result}")

    finally:
        await memory.close()


async def test_multi_tenant_memory():
    """Test multi-tenant memory with different isolation levels."""
    print("\n" + "=" * 60)
    print("Testing Multi-Tenant Memory")
    print("=" * 60)

    config = get_base_config()
    memory = MultiTenantMemory(config)

    try:
        # ===================================
        # User Memory (Persistent)
        # ===================================
        print("\n--- User Memory (Persistent) ---")

        # Alice's preferences
        result = await memory.add(
            "I prefer dark mode and large fonts",
            user_id="alice",
        )
        print(f"Alice's preference added: {result.get('tenant')}")

        # Bob's preferences
        result = await memory.add(
            "I prefer light mode and compact view",
            user_id="bob",
        )
        print(f"Bob's preference added: {result.get('tenant')}")

        # ===================================
        # Agent Memory (Shared Across Users)
        # ===================================
        print("\n--- Agent Memory (Shared) ---")

        # Support bot knowledge
        result = await memory.add(
            "Company policy: Respond to all tickets within 24 hours. "
            "Escalate priority tickets immediately.",
            agent_id="support-bot",
        )
        print(f"Agent knowledge added: {result.get('tenant')}")

        # Sales bot knowledge
        result = await memory.add(
            "Product pricing: Basic plan $10/month, Pro plan $25/month, "
            "Enterprise custom pricing.",
            agent_id="sales-bot",
        )
        print(f"Agent knowledge added: {result.get('tenant')}")

        # ===================================
        # Session Memory (Temporary)
        # ===================================
        print("\n--- Session Memory (Temporary) ---")

        # Current conversation context
        result = await memory.add(
            "User is asking about order #12345. The order was placed on Dec 10. "
            "Status is 'shipped'.",
            session_id="sess-abc-123",
        )
        print(f"Session context added: {result.get('tenant')}")

        # ===================================
        # User + Agent Combination
        # ===================================
        print("\n--- User + Agent Memory ---")

        # Alice's interaction history with support bot
        result = await memory.add(
            "Alice prefers email responses over phone calls. "
            "Her timezone is PST.",
            user_id="alice",
            agent_id="support-bot",
        )
        print(f"User+Agent context added: {result.get('tenant')}")

        # ===================================
        # Search Within Tenant Scope
        # ===================================
        print("\n--- Searching Within Tenant Scopes ---")

        # Search Alice's memories only
        alice_results = await memory.search(
            "What are the user preferences?",
            user_id="alice",
            limit=5,
        )
        print(f"Alice's memories: {len(alice_results.get('results', []))} results")
        for r in alice_results.get("results", []):
            print(f"  - {r.get('memory', '')[:60]}...")

        # Search agent knowledge
        agent_results = await memory.search(
            "What is the pricing?",
            agent_id="sales-bot",
            limit=5,
        )
        print(f"Sales bot knowledge: {len(agent_results.get('results', []))} results")
        for r in agent_results.get("results", []):
            print(f"  - {r.get('memory', '')[:60]}...")

        # Search session context
        session_results = await memory.search(
            "What is the order status?",
            session_id="sess-abc-123",
            limit=5,
        )
        print(f"Session context: {len(session_results.get('results', []))} results")
        for r in session_results.get("results", []):
            print(f"  - {r.get('memory', '')[:60]}...")

    finally:
        await memory.close()


async def test_graph_memory():
    """Test graph memory with Neo4j."""
    print("\n" + "=" * 60)
    print("Testing Graph Memory (Neo4j)")
    print("=" * 60)

    run_neo4j = os.getenv("RUN_NEO4J_DEMO", "false").lower() == "true"
    if not run_neo4j:
        print("Skipping Neo4j demo (set RUN_NEO4J_DEMO=true to enable)")
        return

    config = get_graph_config()
    graph = AsyncGraphMemory(config)

    try:
        # Extract entities and relationships
        result = await graph.add(
            "John works at Google as a software engineer. "
            "He lives in New York and knows Alice who works at Microsoft.",
            user_id="user_123",
        )
        print(f"Extracted entities: {len(result.get('entities', []))}")
        print(f"Extracted relationships: {len(result.get('relationships', []))}")

        # Search for facts
        search_result = await graph.search(
            "Where does John work?",
            user_id="user_123",
            limit=5,
        )
        print(f"Graph search results: {search_result}")

    finally:
        await graph.close()


async def test_multi_tenant_graph_memory():
    """Test multi-tenant graph memory."""
    print("\n" + "=" * 60)
    print("Testing Multi-Tenant Graph Memory")
    print("=" * 60)

    run_neo4j = os.getenv("RUN_NEO4J_DEMO", "false").lower() == "true"
    if not run_neo4j:
        print("Skipping Neo4j demo (set RUN_NEO4J_DEMO=true to enable)")
        return

    config = get_graph_config()
    memory = MultiTenantGraphMemory(config)

    try:
        # Add graph knowledge per agent
        result = await memory.add(
            "The company headquarters is in San Francisco. "
            "The CEO is John Smith. The CTO is Jane Doe.",
            agent_id="knowledge-bot",
        )
        print(f"Agent graph knowledge added: {result.get('tenant')}")
        print(f"  Entities: {len(result.get('entities', []))}")
        print(f"  Relationships: {len(result.get('relationships', []))}")

        # Search agent's graph knowledge
        search_result = await memory.search(
            "Who is the CEO?",
            agent_id="knowledge-bot",
            limit=5,
        )
        print(f"Graph search results: {len(search_result.get('results', []))} facts found")

    finally:
        await memory.close()


async def test_falkordb_memory():
    """Test FalkorDB graph memory (optional)."""
    print("\n" + "=" * 60)
    print("Testing FalkorDB Graph Memory")
    print("=" * 60)

    run_falkor = os.getenv("RUN_FALKORDB_DEMO", "false").lower() == "true"
    if not run_falkor:
        print("Skipping FalkorDB demo (set RUN_FALKORDB_DEMO=true to enable)")
        return

    config = get_falkordb_config()
    graph = AsyncGraphMemory(config)

    try:
        result = await graph.add(
            "I'm John from Google and living in New York",
            user_id="user_123",
        )
        print(f"FalkorDB: Extracted {len(result.get('entities', []))} entities")
        print(f"FalkorDB: Extracted {len(result.get('relationships', []))} relationships")

    finally:
        await graph.close()


async def test_multi_tenant_falkordb_memory():
    """Test multi-tenant graph memory with FalkorDB."""
    print("\n" + "=" * 60)
    print("Testing Multi-Tenant FalkorDB Graph Memory")
    print("=" * 60)

    run_falkor = os.getenv("RUN_FALKORDB_DEMO", "false").lower() == "true"
    if not run_falkor:
        print("Skipping FalkorDB multi-tenant demo (set RUN_FALKORDB_DEMO=true to enable)")
        return

    config = get_falkordb_config()
    memory = MultiTenantGraphMemory(config)

    try:
        # ===================================
        # User-Specific Graph Memory
        # ===================================
        print("\n--- User Graph Memory ---")

        # Alice's knowledge graph
        result = await memory.add(
            "Alice works at Amazon as a data scientist. "
            "She lives in Seattle and collaborates with Bob on ML projects.",
            user_id="alice",
        )
        print(f"Alice's graph added: {result.get('tenant')}")
        print(f"  Entities: {len(result.get('entities', []))}")
        print(f"  Relationships: {len(result.get('relationships', []))}")

        # Bob's knowledge graph
        result = await memory.add(
            "Bob is a machine learning engineer at Meta. "
            "He specializes in NLP and lives in San Francisco.",
            user_id="bob",
        )
        print(f"Bob's graph added: {result.get('tenant')}")
        print(f"  Entities: {len(result.get('entities', []))}")
        print(f"  Relationships: {len(result.get('relationships', []))}")

        # ===================================
        # Agent-Shared Graph Memory
        # ===================================
        print("\n--- Agent Graph Memory (Shared Knowledge) ---")

        # Knowledge bot - company information
        result = await memory.add(
            "TechCorp was founded in 2010 by Sarah Johnson. "
            "The headquarters is in Austin, Texas. "
            "The company has 500 employees and specializes in cloud computing.",
            agent_id="knowledge-bot",
        )
        print(f"Knowledge bot graph added: {result.get('tenant')}")
        print(f"  Entities: {len(result.get('entities', []))}")
        print(f"  Relationships: {len(result.get('relationships', []))}")

        # HR bot - organizational structure
        result = await memory.add(
            "The Engineering department is led by Mike Chen. "
            "The Sales team reports to Lisa Wang. "
            "HR is managed by David Park.",
            agent_id="hr-bot",
        )
        print(f"HR bot graph added: {result.get('tenant')}")
        print(f"  Entities: {len(result.get('entities', []))}")
        print(f"  Relationships: {len(result.get('relationships', []))}")

        # ===================================
        # Session-Specific Graph Memory
        # ===================================
        print("\n--- Session Graph Memory (Temporary Context) ---")

        # Current support session context
        result = await memory.add(
            "Customer John Doe contacted support about order #98765. "
            "The order contains a laptop and was shipped via FedEx. "
            "Expected delivery is December 20, 2025.",
            session_id="support-session-456",
        )
        print(f"Session graph added: {result.get('tenant')}")
        print(f"  Entities: {len(result.get('entities', []))}")
        print(f"  Relationships: {len(result.get('relationships', []))}")

        # ===================================
        # User + Agent Combined Graph Memory
        # ===================================
        print("\n--- User + Agent Graph Memory ---")

        # Alice's specific context with the knowledge bot
        result = await memory.add(
            "Alice is interested in the cloud migration project. "
            "She has completed certifications in AWS and Azure. "
            "Her manager is Mike Chen from Engineering.",
            user_id="alice",
            agent_id="knowledge-bot",
        )
        print(f"Alice + Knowledge bot graph added: {result.get('tenant')}")
        print(f"  Entities: {len(result.get('entities', []))}")
        print(f"  Relationships: {len(result.get('relationships', []))}")

        # ===================================
        # Search Within Different Tenant Scopes
        # ===================================
        print("\n--- Searching Graph Within Tenant Scopes ---")

        # Search Alice's graph
        alice_results = await memory.search(
            "Where does Alice work?",
            user_id="alice",
            limit=5,
        )
        print(f"Alice's graph search: {len(alice_results.get('results', []))} facts found")
        for r in alice_results.get("results", []):
            print(f"  - {r.get('fact', '')[:70]}...")

        # Search knowledge bot's shared graph
        agent_results = await memory.search(
            "Who founded the company?",
            agent_id="knowledge-bot",
            limit=5,
        )
        print(f"Knowledge bot search: {len(agent_results.get('results', []))} facts found")
        for r in agent_results.get("results", []):
            print(f"  - {r.get('fact', '')[:70]}...")

        # Search session context
        session_results = await memory.search(
            "What is the order status?",
            session_id="support-session-456",
            limit=5,
        )
        print(f"Session search: {len(session_results.get('results', []))} facts found")
        for r in session_results.get("results", []):
            print(f"  - {r.get('fact', '')[:70]}...")

        # Search user+agent combined scope
        combined_results = await memory.search(
            "What certifications does Alice have?",
            user_id="alice",
            agent_id="knowledge-bot",
            limit=5,
        )
        print(f"Alice + Knowledge bot search: {len(combined_results.get('results', []))} facts found")
        for r in combined_results.get("results", []):
            print(f"  - {r.get('fact', '')[:70]}...")

        # ===================================
        # Get All Entities for a Tenant
        # ===================================
        print("\n--- Get All Entities by Tenant ---")

        alice_entities = await memory.get_all_entities(user_id="alice", limit=10)
        print(f"Alice's entities: {len(alice_entities.get('entities', []))}")
        for e in alice_entities.get("entities", [])[:3]:
            print(f"  - {e.get('name', '')} ({e.get('entity_type', '')})")

        agent_entities = await memory.get_all_entities(agent_id="knowledge-bot", limit=10)
        print(f"Knowledge bot entities: {len(agent_entities.get('entities', []))}")
        for e in agent_entities.get("entities", [])[:3]:
            print(f"  - {e.get('name', '')} ({e.get('entity_type', '')})")

    finally:
        await memory.close()


async def main():
    """Run all tests."""
    print("=" * 60)
    print("AgenticMemory Multi-Tenant Test Suite")
    print("=" * 60)
    print(f"\nEnvironment Configuration:")
    print(f"  LLM Model: {os.getenv('LLM_MODEL', 'not set')}")
    print(f"  Embedder Model: {os.getenv('EMBEDDER_MODEL', 'not set')}")
    print(f"  PostgreSQL: {os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}")
    print(f"  Qdrant: {os.getenv('QDRANT_URL', 'http://localhost:6333')}")
    print(f"  Redis: {os.getenv('REDIS_URL', 'redis://localhost:6379')}")
    print(f"  Neo4j: {os.getenv('NEO4J_URI', 'bolt://localhost:7687')}")

    # Run tests based on environment configuration
    try:
        # # Basic memory test
        # await test_basic_memory()
        #
        # # Multi-tenant memory test
        # await test_multi_tenant_memory()
        #
        # # Graph memory tests
        # await test_graph_memory()
        # await test_multi_tenant_graph_memory()

        # FalkorDB tests
        await test_falkordb_memory()
        await test_multi_tenant_falkordb_memory()

    except Exception as e:
        print(f"\n❌ Error during tests: {e}")
        raise

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

