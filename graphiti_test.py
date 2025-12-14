import asyncio
import os

from dotenv import load_dotenv
from datetime import datetime
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.nodes import EpisodeType
from graphiti_core.driver.falkordb_driver import FalkorDriver
# Load environment variables
load_dotenv()

async def main():
    llm_client = OpenAIClient(
        config= LLMConfig (
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        ))


    # Create a FalkorDB driver with custom database name
    driver = FalkorDriver(
        host=os.getenv("FALKORDB_HOST", "localhost"),
        port=int(os.getenv("FALKORDB_PORT", "6380")),
        username=os.getenv("FALKORDB_USER", ""),
        password=os.getenv("FALKORDB_PASSWORD", ""),
        database="graphiti_memory",
    )

    # Initialize Graphiti with FalkorDB
    graphiti = Graphiti(graph_driver=driver , llm_client=llm_client)

    # Build indices (run once during setup)
    await graphiti.build_indices_and_constraints()

    # Add an episode (information to be stored in the graph)
    episode_body = """
    Alice met Bob at the AI conference in San Francisco on March 15, 2024.
    They discussed the latest developments in graph databases and decided to 
    collaborate on a new project using FalkorDB.
    """

    await graphiti.add_episode(
        name="Conference Meeting",
        episode_body=episode_body,
        #episode_type=EpisodeType.text,
        reference_time=datetime(2024, 3, 15),
        source_description="Conference notes"
    )

    # Search the knowledge graph
    search_results = await graphiti.search(
        query="What did Alice and Bob discuss?",
        num_results=5
    )

    print("Search Results:")
    for result in search_results:
        print(f"- {result}")

    # Close the connection
    await graphiti.close()


# Run the example
asyncio.run(main())
