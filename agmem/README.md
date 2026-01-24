# AgenticMemory (agmem)

A pluggable, multi-tenant memory framework for AI applications. `agmem` provides a unified interface for Vector Memory and Graph Memory with built-in isolation strategies for users, agents, and sessions.

## Features

- **Vector Memory**: Store and retrieve semantic memories using vector databases (Qdrant).
- **Graph Memory**: Manage knowledge graphs with entities and relationships (Neo4j, FalkorDB).
- **Multi-Tenancy**: Built-in isolation scopes:
  - `User`: Persistent long-term memory for specific users.
  - `Agent`: Shared knowledge base for AI agents.
  - `Session`: Temporary, ephemeral context for conversations.
  - `Combined`: Intersection of scopes (e.g., User's history with a specific Agent).
- **Pluggable Architecture**: Easily swap LLM providers (OpenAI, etc.) and storage backends.

## Installation

```bash
pip install agmem
```

## Configuration

`agmem` uses a configuration dictionary or environment variables. Standard environment variables include:

```env
# LLM & Embeddings
LLM_MODEL=gpt-4-turbo
LLM_API_KEY=sk-...
EMBEDDER_MODEL=text-embedding-3-small
EMBEDDER_API_KEY=sk-...

# Vector Store (Qdrant)
QDRANT_URL=http://localhost:6333

# Graph Store (Neo4j)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Storage (Postgres)
POSTGRES_HOST=localhost
POSTGRES_DB=agentic_memory
```

## Usage

### 1. Basic Vector Memory

```python
import asyncio
from agmem import AsyncMemory

config = {
    "llm": {"provider": "openai", "config": {...}},
    "vector_store": {"provider": "qdrant", "config": {"url": "http://localhost:6333"}}
}

async def main():
    memory = AsyncMemory(config)
    
    # Add a memory
    await memory.add(
        "I prefer dark mode and use Python.",
        user_id="user_123"
    )
    
    # Search
    results = await memory.search(
        "What are my preferences?",
        user_id="user_123"
    )
    print(results)
    
    await memory.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Multi-Tenant Memory

Isolate memory contexts by `user_id`, `agent_id`, or `session_id`.

```python
from agmem import MultiTenantMemory

async def multi_tenant_example():
    memory = MultiTenantMemory(config)
    
    # 1. User Memory (Persistent)
    await memory.add("My name is Alice.", user_id="alice")
    
    # 2. Agent Knowledge (Shared)
    await memory.add(
        "Support Policy: Reply within 24h.", 
        agent_id="support-bot"
    )
    
    # 3. Session Context (Ephemeral)
    await memory.add(
        "User is asking about order #555.", 
        session_id="session_abc"
    )
    
    # Search specific scope
    # Find what the support bot knows about policies
    policy = await memory.search(
        "What is the reply time?", 
        agent_id="support-bot"
    )
```

### 3. Graph Memory

Manage structured knowledge with entities and relationships.

```python
from agmem import AsyncGraphMemory

async def graph_example():
    graph = AsyncGraphMemory(config)
    
    # Extract entities and relationships automatically
    await graph.add(
        "Elon Musk is the CEO of Tesla and SpaceX.",
        user_id="public_news"
    )
    
    # Search for structured facts
    facts = await graph.search("Who is the CEO of Tesla?", user_id="public_news")
    # Result: "Elon Musk is CEO of Tesla"
    
    await graph.close()
```

## Core Modules

- `agmem.memory`: Vector memory implementation.
- `agmem.graph`: Graph memory implementation.
- `agmem.multi_tenant`: High-level wrapper for scoping memories.
