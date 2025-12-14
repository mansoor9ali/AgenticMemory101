"""
Hotel Management AI Agent with Multi-Tenant Graph Memory

This shows how to integrate MultiTenantGraphMemory (FalkorDB) with an AI agent
that handles hotel guest interactions.

Features:
- Multi-tenant memory isolation per guest
- Real LLM integration for intelligent responses
- Context-aware conversations
- Guest preference tracking
- Interaction history
"""

import asyncio
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

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
            "graph_name": "hotel_memory",
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


def get_llm_config() -> dict:
    """Get LLM configuration for response generation."""
    return {
        "model": os.getenv("LLM_MODEL"),
        "base_url": os.getenv("LLM_BASE_URL"),
        "api_key": os.getenv("LLM_API_KEY"),
    }


async def generate_agent_response(
    system_prompt: str,
    user_message: str,
    use_real_llm: bool = True
) -> str:
    """
    Generate AI agent response using LLM.

    Args:
        system_prompt: The system prompt with context
        user_message: The user's message
        use_real_llm: Whether to use real LLM or simulated response

    Returns:
        The AI response string
    """
    if not use_real_llm:
        # Simulated response for testing
        return "I understand your request. Based on your preferences, I'll help you with that."

    try:
        from openai import OpenAI

        config = get_llm_config()
        client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )

        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=500,
            temperature=0.7,
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"⚠️ LLM call failed: {e}")
        return "I apologize, but I'm having trouble processing your request. How else may I assist you?"


def format_context_from_search(search_results: Dict[str, Any]) -> str:
    """Format search results into a context string for the AI agent."""
    facts = search_results.get("results", [])
    if not facts:
        return "No previous context available for this guest."

    context_lines = []
    for fact in facts:
        fact_text = fact.get("fact", "")
        if fact_text:
            context_lines.append(f"- {fact_text}")

    return "\n".join(context_lines) if context_lines else "No previous context available."


class HotelManagementAgent:
    """
    Hotel Management AI Agent with Multi-Tenant Graph Memory.

    Uses FalkorDB-backed graph memory to store and retrieve guest information,
    preferences, and interaction history.

    Features:
    - Memory isolation per guest (multi-tenancy)
    - Contextual conversation with memory recall
    - Guest preference learning
    - Real LLM integration
    """

    def __init__(self, use_real_llm: bool = True):
        """
        Initialize the Hotel Management Agent.

        Args:
            use_real_llm: Whether to use real LLM for responses (default: True)
        """
        self.config = get_falkordb_config()
        self.memory: Optional[MultiTenantGraphMemory] = None
        self.use_real_llm = use_real_llm
        self.hotel_name = "Grand Azure Hotel"

    async def initialize(self):
        """Initialize the memory system."""
        self.memory = MultiTenantGraphMemory(self.config)
        print(f"✅ {self.hotel_name} AI Concierge initialized with FalkorDB memory")

    async def close(self):
        """Close all connections."""
        if self.memory:
            await self.memory.close()
            print("✅ Memory connections closed")

    async def add_interaction(
        self,
        messages: str | List[Dict[str, str]],
        guest_id: str,
    ) -> Dict[str, Any]:
        """
        Add a guest interaction to memory.

        Args:
            messages: Either a string or list of message dicts with 'role' and 'content'
            guest_id: The unique guest identifier

        Returns:
            Result dict with entities and relationships extracted
        """
        # Convert message list to a formatted string for better entity extraction
        if isinstance(messages, list):
            content = "\n".join([
                f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
                for msg in messages
            ])
        else:
            content = messages

        result = await self.memory.add(content, user_id=guest_id)
        return result

    async def get_context(
        self,
        query: str,
        guest_id: str,
        limit: int = 10,
    ) -> str:
        """
        Get relevant context for a guest query.

        Args:
            query: The current query/message from the guest
            guest_id: The unique guest identifier
            limit: Maximum number of facts to retrieve

        Returns:
            Formatted context string for the AI agent
        """
        search_results = await self.memory.search(
            query=query,
            user_id=guest_id,
            limit=limit,
        )
        return format_context_from_search(search_results)

    async def get_all_guest_info(
        self,
        guest_id: str,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Get all stored information about a guest.

        Args:
            guest_id: The unique guest identifier
            limit: Maximum number of entities to retrieve

        Returns:
            Dict with all entities and relationships for the guest
        """
        entities = await self.memory.get_all_entities(user_id=guest_id, limit=limit)
        relationships = await self.memory.get_all_relationships(user_id=guest_id, limit=limit)

        return {
            "entities": entities.get("entities", []),
            "relationships": relationships.get("relationships", []),
        }

    def build_system_prompt(self, context: str, guest_name: str = "Guest") -> str:
        """
        Build the system prompt for the AI agent with guest context.

        Args:
            context: The formatted context string
            guest_name: The guest's name if known

        Returns:
            Complete system prompt for the AI agent
        """
        current_time = datetime.now().strftime("%I:%M %p")
        current_date = datetime.now().strftime("%B %d, %Y")

        return f"""You are the AI concierge for {self.hotel_name}. Current time: {current_time}, {current_date}.

## Guest Information
{context}

## Your Capabilities
- Room service ordering (available 6 AM - 11 PM)
- Restaurant reservations
- Spa and wellness bookings
- Local recommendations
- Transportation arrangements
- Housekeeping requests
- Wake-up calls
- Billing inquiries

## Instructions
- Be warm, professional, and personalized
- Address the guest by name when known
- Reference their known preferences proactively
- Suggest relevant services based on their history
- If they previously liked something, recommend similar options
- Keep responses concise but helpful
- Always confirm requests before processing
- For dining, consider any dietary restrictions mentioned previously"""

    async def handle_guest_interaction(
        self,
        guest_id: str,
        message: str,
        guest_name: str = "Guest",
    ) -> str:
        """
        Handle a complete guest interaction cycle.

        1. Retrieve relevant context from memory
        2. Build system prompt with context
        3. Generate AI response
        4. Store the interaction in memory

        Args:
            guest_id: The unique guest identifier
            message: The guest's message
            guest_name: The guest's name for personalization

        Returns:
            The AI agent's response
        """
        # Get context for the agent
        context = await self.get_context(message, guest_id)

        # Build system prompt with context
        system_prompt = self.build_system_prompt(context, guest_name)

        # Generate response using LLM
        response = await generate_agent_response(
            system_prompt,
            message,
            use_real_llm=self.use_real_llm
        )

        # Store the new interaction
        await self.add_interaction([
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ], guest_id)

        return response


async def hotel_agent_demo():
    """
    Demonstration of the Hotel Management AI Agent.

    Shows how the agent handles guest interactions with memory persistence,
    including multi-turn conversations and context retrieval.
    """
    # Set to False for simulated responses, True for real LLM calls
    USE_REAL_LLM = True

    agent = HotelManagementAgent(use_real_llm=USE_REAL_LLM)
    guest_id = "guest-maria-garcia"
    guest_name = "Maria Garcia"

    try:
        await agent.initialize()

        # =============================================
        # Phase 1: Setting up guest history
        # =============================================
        print("\n" + "=" * 60)
        print("📝 PHASE 1: Recording Guest History")
        print("=" * 60)

        # Check-in with room details
        print("\n1️⃣ Check-in interaction...")
        result1 = await agent.add_interaction([
            {"role": "user", "content": "Hi, I'm Maria Garcia. I just arrived for my reservation."},
            {"role": "assistant", "content": "Welcome to Grand Azure Hotel, Ms. Garcia! Your ocean-view suite, room 501, is ready. I've noted your preference for a high floor. May I arrange for your luggage to be brought up?"},
        ], guest_id)
        print(f"   ✓ Extracted: {len(result1.get('entities', []))} entities, "
              f"{len(result1.get('relationships', []))} relationships")

        # Guest preferences
        print("\n2️⃣ Recording preferences...")
        result2 = await agent.add_interaction(
            "I have some specific preferences: I need the room temperature at 68°F, "
            "extra towels, and I'm allergic to shellfish. Also, I prefer sparkling water "
            "in the minibar instead of still water.",
            guest_id
        )
        print(f"   ✓ Extracted: {len(result2.get('entities', []))} entities, "
              f"{len(result2.get('relationships', []))} relationships")

        # Restaurant recommendation
        print("\n3️⃣ Recording restaurant inquiry...")
        result3 = await agent.add_interaction([
            {"role": "user", "content": "Can you recommend a good Italian restaurant nearby? I love authentic pasta dishes."},
            {"role": "assistant", "content": "I highly recommend Bella Vista, just a 5-minute walk from the hotel. They're known for their handmade pasta and have excellent vegetarian options. Shall I make a reservation for you?"},
        ], guest_id)
        print(f"   ✓ Extracted: {len(result3.get('entities', []))} entities, "
              f"{len(result3.get('relationships', []))} relationships")

        # Spa booking
        print("\n4️⃣ Recording spa booking...")
        result4 = await agent.add_interaction([
            {"role": "user", "content": "I'd like to book a massage for tomorrow morning, around 10 AM if possible."},
            {"role": "assistant", "content": "I've booked you a 60-minute Swedish massage at our Azure Spa for tomorrow at 10 AM. The spa is located on the 3rd floor. Would you like me to add any other treatments?"},
        ], guest_id)
        print(f"   ✓ Extracted: {len(result4.get('entities', []))} entities, "
              f"{len(result4.get('relationships', []))} relationships")

        # =============================================
        # Phase 2: New Multi-Turn Conversation
        # =============================================
        print("\n" + "=" * 60)
        print("💬 PHASE 2: New Conversation (with Memory Context)")
        print("=" * 60)

        conversations = [
            "I'd like to order room service for dinner tonight. What do you recommend?",
            "That sounds good. I'll have the pasta. Can you also send up some sparkling water?",
            "What time is my spa appointment tomorrow again?",
        ]

        for i, message in enumerate(conversations, 1):
            print(f"\n{'─' * 50}")
            print(f"🧑 Guest [{i}]: {message}")

            # Get context being used
            context = await agent.get_context(message, guest_id)
            print(f"\n📋 Memory Context Retrieved:")
            for line in context.split('\n')[:3]:  # Show first 3 context items
                print(f"   {line}")
            if context.count('\n') > 3:
                print(f"   ... and {context.count(chr(10)) - 2} more")

            # Get AI response
            response = await agent.handle_guest_interaction(guest_id, message, guest_name)
            print(f"\n🤖 Concierge: {response}")

        # =============================================
        # Phase 3: Guest Memory Summary
        # =============================================
        print("\n" + "=" * 60)
        print("🧠 PHASE 3: Guest Memory Knowledge Graph")
        print("=" * 60)

        guest_info = await agent.get_all_guest_info(guest_id)

        print(f"\n📊 Entities ({len(guest_info['entities'])} total):")
        for entity in guest_info['entities'][:8]:
            print(f"   • {entity.get('name', 'N/A'):20} [{entity.get('entity_type', 'N/A')}]")
        if len(guest_info['entities']) > 8:
            print(f"   ... and {len(guest_info['entities']) - 8} more")

        print(f"\n🔗 Relationships ({len(guest_info['relationships'])} total):")
        for rel in guest_info['relationships'][:8]:
            fact = rel.get('fact', 'N/A')
            print(f"   • {fact[:65]}{'...' if len(fact) > 65 else ''}")
        if len(guest_info['relationships']) > 8:
            print(f"   ... and {len(guest_info['relationships']) - 8} more")

    finally:
        await agent.close()


async def interactive_demo():
    """
    Interactive demo where you can chat with the hotel agent.
    """
    agent = HotelManagementAgent(use_real_llm=True)
    guest_id = "guest-interactive"
    guest_name = "Guest"

    try:
        await agent.initialize()

        print("\n" + "=" * 60)
        print("🏨 Welcome to Grand Azure Hotel Interactive Demo!")
        print("=" * 60)
        print("\nType your message to chat with the AI concierge.")
        print("Commands: 'quit' to exit, 'memory' to see stored info\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() == 'quit':
                print("\nThank you for staying with us! Goodbye.")
                break
            if user_input.lower() == 'memory':
                info = await agent.get_all_guest_info(guest_id)
                print(f"\n📊 Your Memory: {len(info['entities'])} entities, "
                      f"{len(info['relationships'])} relationships\n")
                continue

            response = await agent.handle_guest_interaction(guest_id, user_input, guest_name)
            print(f"Concierge: {response}\n")

    finally:
        await agent.close()


async def main():
    """Main entry point."""
    print("\n" + "🏨" * 25)
    print("   HOTEL MANAGEMENT AI AGENT")
    print("   with Multi-Tenant Graph Memory (FalkorDB)")
    print("🏨" * 25 + "\n")

    # Choose demo mode
    print("Select demo mode:")
    print("  1. Automated demo (shows memory features)")
    print("  2. Interactive chat mode")

    try:
        choice = input("\nEnter choice (1/2) [default: 1]: ").strip() or "1"
    except EOFError:
        choice = "1"

    if choice == "2":
        await interactive_demo()
    else:
        await hotel_agent_demo()

    print("\n✅ Hotel Agent Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())

