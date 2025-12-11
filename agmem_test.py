from agmem import AsyncMemory, AsyncGraphMemory
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def main():
    # Vector memory
    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": os.getenv("LLM_MODEL"),
                "base_url": os.getenv("LLM_BASE_URL"),
                "api_key": os.getenv("LLM_API_KEY")
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": os.getenv("EMBEDDER_MODEL"),
                "base_url": os.getenv("EMBEDDER_BASE_URL"),
                "api_key": os.getenv("EMBEDDER_API_KEY")
            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "url": os.getenv("QDRANT_URL")
            }
        },
        "storage": {
            "provider": "postgres",
            "config": {
                "host": os.getenv("POSTGRES_HOST"),
                "port": os.getenv("POSTGRES_PORT"),
                "database": os.getenv("POSTGRES_DATABASE"),
                "user": os.getenv("POSTGRES_USER"),
                "password": os.getenv("POSTGRES_PASSWORD")
            }
        },
        "cache": {
            "redis_url": os.getenv("REDIS_URL"),
            "enabled": True
        },
    }
    memory = AsyncMemory(config)
    await memory.add("I prefer dark mode", user_id="user_123")
    await memory.add("Hi My Name is John, living in New york and my user_id is user_123", user_id="user_123")
    await memory.close()

    # Graph memory

    # config2 = {
    #     "graph_store": {
    #         "provider": "neo4j",
    #         "uri": os.getenv("NEO4J_URI"),
    #         "user": os.getenv("NEO4J_USER"),
    #         "password": os.getenv("NEO4J_PASSWORD"),
    #     },
    #     "llm": {
    #         "provider": "openai",
    #         "config": {
    #             "model": os.getenv("LLM_MODEL"),
    #             "base_url": os.getenv("LLM_BASE_URL"),
    #             "api_key": os.getenv("LLM_API_KEY")
    #         }
    #     },
    #     "embedder": {
    #         "provider": "openai",
    #         "config": {
    #             "model": os.getenv("EMBEDDER_MODEL"),
    #             "base_url": os.getenv("EMBEDDER_BASE_URL"),
    #             "api_key": os.getenv("EMBEDDER_API_KEY")
    #         }
    #     },
    #     "max_entities_per_message": 10,
    #     "max_relationships_per_message": 15,
    # }
    # graph = AsyncGraphMemory(config2)
    # await graph.add("I'm John from Google", user_id="user_123")
    # await graph.close()

if __name__ == '__main__':
    asyncio.run(main())