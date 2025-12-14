import asyncio
import os

from dotenv import load_dotenv
from agmem import MultiTenantGraphMemory

# Load environment variables
load_dotenv()


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


# Initialize Multi-Tenant Graph Memory
# Configure FalkorDB connection
config = get_falkordb_config()


async def main():
    """Main async function to run the multi-tenant graph memory test."""
    print("Starting Multi-Tenant Graph Memory Test...")

    # Vector Memory with Multi-Tenancy
    memory = MultiTenantGraphMemory(config)

    try:
        # User memory (persistent)
        print("\n1. Adding user memory for 'alice'...")
        result1 = await memory.add("I prefer dark mode", user_id="alice")
        print(f"   Added: {len(result1.get('entities', []))} entities, {len(result1.get('relationships', []))} relationships")

        # Agent memory (shared across users)
        print("\n2. Adding agent memory for 'support-bot'...")
        result2 = await memory.add("Company policy: respond within 24 hours", agent_id="support-bot")
        print(f"   Added: {len(result2.get('entities', []))} entities, {len(result2.get('relationships', []))} relationships")

        # Session memory (temporary)
        print("\n3. Adding session memory for 'sess-123'...")
        result3 = await memory.add("Currently discussing order #12345", session_id="sess-123")
        print(f"   Added: {len(result3.get('entities', []))} entities, {len(result3.get('relationships', []))} relationships")

        # User + Agent combination
        print("\n4. Adding user+agent memory for 'alice' + 'support-bot'...")
        result4 = await memory.add("Alice prefers email", user_id="alice", agent_id="support-bot")
        print(f"   Added: {len(result4.get('entities', []))} entities, {len(result4.get('relationships', []))} relationships")

        # Search within tenant scope
        print("\n5. Searching for 'preferences' in alice's memory...")
        results = await memory.search("preferences", user_id="alice")

        print(f"\nSearch Results ({len(results.get('results', []))} facts found):")
        for r in results.get("results", []):
            print(f"   - {r.get('fact', '')[:80]}...")

        print(f"\nTenant Info: {results.get('tenant')}")

    finally:
        await memory.close()
        print("\n✅ Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
