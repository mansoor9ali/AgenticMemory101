import asyncio
import os
import logging
from dotenv import load_dotenv

from agmem import AsyncMemory
from agmem.processing.fact_extractor import FactExtractionProcessor
from agmem.processing.profiler import UserProfilingProcessor
from agmem.processing.event_detector import EventDetectionProcessor
from agmem.scoring.hybrid import HybridScorer

# Load env variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CognitiveTest")

def get_config():
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": os.getenv("LLM_MODEL", "gpt-4-turbo"),
                "api_key": os.getenv("LLM_API_KEY"),
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": os.getenv("EMBEDDER_MODEL", "text-embedding-3-small"),
                "api_key": os.getenv("EMBEDDER_API_KEY"),
            },
        },
        "vector_store": {"provider": "qdrant", "config": {"url": os.getenv("QDRANT_URL", "http://localhost:6333")}},
        "storage": {"provider": "postgres", "config": { ... }} # Assuming standard config
    }

async def main():
    print("🔹 Starting Cognitive Pipeline Test...")
    
    # 1. Initialize with Processors
    # We need to manually init LLM for processors first, or let AsyncMemory handle it if we pass classes?
    # Current implementation expects INITIALIZED services in processors. 
    # Let's fix this in the test by using a temporary AsyncMemory to get the LLM factory working, 
    # or just use the factory directly.
    
    from agmem.utils.factory import LLMFactory
    
    llm = LLMFactory.create("openai", {
        "model": os.getenv("LLM_MODEL", "gpt-4-turbo"),
        "api_key": os.getenv("LLM_API_KEY")
    })
    
    processors = [
        FactExtractionProcessor(llm),
        UserProfilingProcessor(llm),
        EventDetectionProcessor(llm)
    ]
    
    scorer = HybridScorer(weight_semantic=0.8, weight_importance=0.1, weight_keyword=0.1)
    
    # Note: reusing base config from environment
    config = {
         "llm": {"provider": "openai", "config": {"api_key": os.getenv("LLM_API_KEY")}},
         "embedder": {"provider": "openai", "config": {"api_key": os.getenv("EMBEDDER_API_KEY")}},
         "vector_store": {"provider": "qdrant", "config": {"url": "http://localhost:6333"}},
         "storage": {
            "provider": "postgres", 
            "config": {
                "host": os.getenv("POSTGRES_HOST", "localhost"),
                "port": int(os.getenv("POSTGRES_PORT", "5432")),
                "database": os.getenv("POSTGRES_DATABASE", "agenticMermoryDB"),
                "user": os.getenv("POSTGRES_USER", "postgres"),
                "password": os.getenv("POSTGRES_PASSWORD", "password"),
            }
        },
    }
    
    memory = AsyncMemory(config, processors=processors, scorer=scorer)
    
    try:
        user_id = "test_user_cognitive"
        
        # 2. Add Complex Input (Should trigger processors)
        print("\nTest 1: Adding complex cognitive input...")
        text = """
        I'm heavily allergic to peanuts. 
        I have a quarterly review meeting with my boss next Tuesday at 2 PM.
        I prefer concise, bullet-point summaries.
        """
        
        res = await memory.add(text, user_id=user_id)
        print(f"Added {len(res.get('results', []))} memories.")
        for r in res.get('results', []):
            print(f" - [{r.get('event')}] {r.get('memory')[:60]}...")
            
        # 3. Add Raw Input (Should skip processors)
        print("\nTest 2: Adding raw input (skip_processing=True)...")
        raw_text = "Raw system log 9999"
        res_raw = await memory.add(raw_text, user_id=user_id, skip_processing=True)
        print(f"Added {len(res_raw.get('results', []))} memories.")
        for r in res_raw.get('results', []):
            print(f" - [{r.get('event')}] {r.get('memory')}")
            
        # 4. Search with Hybrid Scorer
        print("\nTest 3: Hybrid Search...")
        search_res = await memory.search("allergies and meetings", user_id=user_id)
        for r in search_res.get('results', []):
            print(f" - Score: {r['score']:.3f} | {r['memory'][:60]}...")
            
    finally:
        await memory.close()
        print("\n✅ Test Completed")

if __name__ == "__main__":
    if not os.getenv("LLM_API_KEY"):
        print("Skipping test: LLM_API_KEY is missing")
    else:
        asyncio.run(main())
