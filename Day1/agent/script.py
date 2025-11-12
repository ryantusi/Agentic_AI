"""
Your First AI Agent with ADK
A script to build and run an AI agent using Google's Agent Development Kit
"""

import os
import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types


def setup_environment():
    """Load environment variables and configure API key"""
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "🔑 GOOGLE_API_KEY not found in environment variables.\n"
            "Please create a .env file with: GOOGLE_API_KEY=your_api_key_here"
        )
    
    os.environ["GOOGLE_API_KEY"] = api_key
    print("✅ Gemini API key setup complete.")
    return api_key


def create_retry_config():
    """Configure retry options for handling transient errors"""
    retry_config = types.HttpRetryOptions(
        attempts=5,  # Maximum retry attempts
        exp_base=7,  # Delay multiplier
        initial_delay=1,  # Initial delay before first retry (in seconds)
        http_status_codes=[429, 500, 503, 504]  # Retry on these HTTP errors
    )
    print("✅ Retry configuration created.")
    return retry_config


def create_agent(retry_config):
    """Define and configure the AI agent"""
    root_agent = Agent(
        name="helpful_assistant",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        description="A simple agent that can answer general questions.",
        instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
        tools=[google_search],
    )
    print("✅ Agent defined successfully.")
    return root_agent


def create_runner(agent):
    """Create a runner to orchestrate the agent"""
    runner = InMemoryRunner(agent=agent)
    print("✅ Runner created successfully.")
    return runner


async def run_agent_query(runner, query):
    """Run a query through the agent and display results"""
    print(f"\n{'='*80}")
    print(f"🤔 Query: {query}")
    print(f"{'='*80}\n")
    
    response = await runner.run_debug(query)
    
    print(f"\n{'='*80}")
    print("✅ Query completed!")
    print(f"{'='*80}\n")
    
    return response


async def interactive_mode(runner):
    """Run the agent in interactive mode"""
    print("\n" + "="*80)
    print("🎯 INTERACTIVE MODE")
    print("="*80)
    print("Ask your agent questions! Type 'exit', 'quit', or 'q' to stop.")
    print("="*80 + "\n")
    
    while True:
        try:
            query = input("Your question: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q', '']:
                print("\n👋 Goodbye!")
                break
            
            await run_agent_query(runner, query)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


async def run_predefined_queries(runner):
    """Run some example queries"""
    queries = [
        "What is Agent Development Kit from Google? What languages is the SDK available in?",
        "What's the weather in London today?",
        "Who won the last soccer world cup?"
    ]
    
    print("\n" + "="*80)
    print("🚀 RUNNING PREDEFINED QUERIES")
    print("="*80 + "\n")
    
    for query in queries:
        await run_agent_query(runner, query)
        print("\n" + "-"*80 + "\n")
        await asyncio.sleep(2)  # Brief pause between queries


async def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("🤖 AI AGENT WITH ADK - GETTING STARTED")
    print("="*80 + "\n")
    
    try:
        # Setup
        setup_environment()
        retry_config = create_retry_config()
        agent = create_agent(retry_config)
        runner = create_runner(agent)
        
        print("\n" + "="*80)
        print("✅ ALL COMPONENTS INITIALIZED SUCCESSFULLY!")
        print("="*80)
        
        # Choose mode
        print("\nSelect mode:")
        print("1. Run predefined example queries")
        print("2. Interactive mode (ask your own questions)")
        print("3. Run both")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == "1":
            await run_predefined_queries(runner)
        elif choice == "2":
            await interactive_mode(runner)
        elif choice == "3":
            await run_predefined_queries(runner)
            await interactive_mode(runner)
        else:
            print("Invalid choice. Running predefined queries...")
            await run_predefined_queries(runner)
        
        print("\n" + "="*80)
        print("🎉 SCRIPT COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}\n")
        raise


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
