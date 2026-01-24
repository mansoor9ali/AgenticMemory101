"""
FinanceGraphMemoryDemo.py

A comprehensive example of using AgenticMemory's MultiTenantGraphMemory
within the Financial Services domain.

Scenarios Demonstrated:
1.  **User Scope (Investor Profile)**: Persistent memory for an investor's risk tolerance, portfolio, and goals.
2.  **Agent Scope (Specialized Bots)**: Shared knowledge bases for Market Analysis and Compliance.
3.  **Session Scope (Event Context)**: Temporary context for a specific earnings call or meeting.
4.  **Combined Scope**: Intersection of user specifics and agent knowledge.

Prerequisites:
- A running Graph Database (Neo4j or FalkorDB)
- OpenAI API Key (for LLM and Embeddings)
"""

import asyncio
import os
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

# Import AgenticMemory components
from agmem import MultiTenantGraphMemory

# Init logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FinanceDemo")

# Load environment variables
load_dotenv()

def get_config() -> Dict[str, Any]:
    """
    Constructs the configuration for AgenticMemory.
    Defaults to FalkorDB for this demo, but can be switched to Neo4j.
    """
    
    # Check if we should use Neo4j or FalkorDB based on env, default to FalkorDB
    use_neo4j = os.getenv("USE_NEO4J", "false").lower() == "true"
    
    graph_store_config = {}
    if use_neo4j:
        graph_store_config = {
            "provider": "neo4j",
            "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "user": os.getenv("NEO4J_USER", "neo4j"),
            "password": os.getenv("NEO4J_PASSWORD", "password"),
        }
    else:
        graph_store_config = {
            "provider": "falkordb",
            "host": os.getenv("FALKORDB_HOST", "localhost"),
            "port": int(os.getenv("FALKORDB_PORT", "6380")),
            "password": os.getenv("FALKORDB_PASSWORD", ""),
        }

    return {
        "graph_store": graph_store_config,
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
        # Tuning extraction for finance density
        "max_entities_per_message": 20,
        "max_relationships_per_message": 30,
    }

async def print_results(section: str, result: Dict[str, Any]):
    """Helper to print results nicely."""
    print(f"\n{'='*60}")
    print(f"🔹 {section}")
    print(f"{'='*60}")
    
    tenant = result.get('tenant', {})
    print(f"Tenant Scope: [{tenant.get('tenant_type', 'UNKNOWN').upper()}] {tenant.get('tenant_id', 'N/A')}")
    
    if 'entities' in result:
        print(f"\n📥 Extracted Data:")
        print(f"   - Entities Found: {len(result['entities'])}")
        print(f"   - Relationships Found: {len(result['relationships'])}")
        
    if 'results' in result:
        print(f"\n🔍 Search Results ({len(result['results'])}):")
        for i, item in enumerate(result['results'], 1):
            fact = item.get('fact', item.get('memory', ''))
            print(f"   {i}. {fact}")
            
    print("-" * 60)

async def main():
    logger.info("Starting Finance Graph Memory Demo...")
    
    # 1. Initialize Memory
    config = get_config()
    try:
        memory = MultiTenantGraphMemory(config)
    except Exception as e:
        logger.error(f"Failed to initialize memory: {e}")
        return

    try:
        # =========================================================================
        # SCENARIO 1: USER SCOPE - Onboarding an Investor
        # =========================================================================
        # Alice is a new High Net Worth Individual (HNWI) client.
        # We store her profile in a persistent User scope.
        
        user_id = "investor_alice_88"
        
        logger.info(f"Processing Scenario 1: User Profile for {user_id}")
        
        alice_profile_text = """
        Alice creates a new investment account. 
        She classifies her risk tolerance as Aggressive. 
        She is specifically interested in the Technology and Renewable Energy sectors.
        She currently holds $500,000 in NVDA (Nvidia) and has a negative sentiment towards fossil fuel companies.
        She wants to retire by 2040 with a target portfolio value of $10M.
        """
        
        res_user = await memory.add(alice_profile_text, user_id=user_id)
        await print_results("Scenario 1: Investor Profile Creation", res_user)

        # =========================================================================
        # SCENARIO 2: AGENT SCOPE - Specialized Knowledge Bases
        # =========================================================================
        # We have two specialized agents: a Market Analyst and a Compliance Officer.
        # Their knowledge is shared across all users but isolated from each other.

        agent_market = "agent_market_analyst_01"
        agent_compliance = "agent_compliance_officer_01"

        logger.info("Processing Scenario 2: Agent Knowledge Ingestion")

        # 2a. Market Analyst Agent Knowledge
        market_intel_text = """
        The Technology sector is currently experiencing a boom driven by Generative AI.
        Nvidia (NVDA) dominates the GPU market with 80% market share.
        Regulatory headwinds in the EU might impact Big Tech valuations in Q3 2025.
        Renewable Energy stocks are volatile due to fluctuating interest rates.
        """
        res_agent_market = await memory.add(market_intel_text, agent_id=agent_market)
        await print_results("Scenario 2a: Market Analyst Knowledge", res_agent_market)

        # 2b. Compliance Agent Knowledge
        compliance_text = """
        SEC Rule 10b-5 prohibits employment of manipulative and deceptive devices.
        Investments exceeding $10,000 must be reported for AML (Anti-Money Laundering) checks.
        GDPR requires strict data privacy for all EU-based investors.
        """
        res_agent_compliance = await memory.add(compliance_text, agent_id=agent_compliance)
        await print_results("Scenario 2b: Compliance Officer Knowledge", res_agent_compliance)

        # =========================================================================
        # SCENARIO 3: SESSION SCOPE - Live Earnings Call Context
        # =========================================================================
        # Alice is listening to a specific live earnings call. This data is ephemeral
        # and relevant only for this specific session context.
        
        session_id = "session_earnings_call_q3_2025_nvda"
        
        logger.info(f"Processing Scenario 3: Session Context for {session_id}")
        
        earnings_call_notes = """
        Nvidia CEO Jensen Huang just announced Q3 revenue of $35B, beating estimates by 10%.
        Data Center revenue grew by 150% YoY.
        However, gross margins slightly compressed due to supply chain constraints.
        Guidance for Q4 is conservative.
        """
        res_session = await memory.add(earnings_call_notes, session_id=session_id)
        await print_results("Scenario 3: Live Earnings Call Notes", res_session)

        # =========================================================================
        # SCENARIO 4: QUERIES & COMBINED SCOPES
        # =========================================================================
        logger.info("Processing Scenario 4: Complex Queries")

        # Query 1: What is Alice's financial goal? (User Scope)
        # -------------------------------------------------------------------------
        q1 = await memory.search("What are Alice's retirement goals and holdings?", user_id=user_id)
        await print_results("Query 1 [User Scope]: Alice's Profile", q1)

        # Query 2: What is the market outlook for her holdings? (User + Agent Scope)
        # We perform two searches: 
        #   1. Find Alice's holdings (User Scope) -> "NVDA"
        #   2. Ask the Market Agent about those holdings (Agent Scope)
        # This simulates a "Combined" reasoning step which the application logic would handle.
        # Alternatively, we can use the 'Combined' tenant type if we stored data that way, 
        # but here we demonstrate cross-referencing.
        # -------------------------------------------------------------------------
        
        print("\n🤔 Reasoning: Checking Market Intel for Alice's Holdings...")
        
        # A. Get Agent's view on Tech/NVDA
        q2_agent = await memory.search("What is the outlook for Nvidia and Tech sector?", agent_id=agent_market)
        await print_results("Query 2a [Agent Scope]: Market Analyst Intel", q2_agent)

        # Query 3: Compliance Check (Combined Context)
        # Alice asks: "Can I buy $20,000 more of NVDA right now?"
        # We need to check Compliance Agent rules.
        # -------------------------------------------------------------------------
        q3_compliance = await memory.search("What are the reporting requirements for large investments?", agent_id=agent_compliance)
        await print_results("Query 3 [Agent Scope]: Compliance Check", q3_compliance)

        # Query 4: Session Specifics
        # Alice asks: "Did they mention margins in the call?"
        # -------------------------------------------------------------------------
        q4_session = await memory.search("What was said about margins?", session_id=session_id)
        await print_results("Query 4 [Session Scope]: Earnings Call Details", q4_session)

    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        raise
    finally:
        logger.info("Closing memory connections...")
        await memory.close()
        logger.info("Done.")

if __name__ == "__main__":
    # Check for keys before running
    if not os.getenv("LLM_API_KEY"):
        print("❌ SKIPPING: LLM_API_KEY not found in environment variables.")
        print("Please set LLM_API_KEY in .env to run this demo.")
    else:
        asyncio.run(main())
