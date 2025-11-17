"""
🚀 Agent Tools (Kaggle Day 2)

A script to demonstrate custom tools, code execution, and multi-agent
tool use (Agent-as-a-Tool) with the Google Agent Development Kit (ADK).

This script covers:
- Creating custom Python functions as tools.
- Using clear docstrings and type hints for the LLM.
- Returning structured error messages.
- Creating a specialist agent (for calculations).
- Using that specialist agent as a tool in another agent.
"""

import os
import asyncio
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool
from google.adk.code_executors import BuiltInCodeExecutor

# ---
# 1. Environment and Configuration
# ---

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

# ---
# 2. Custom Function Tools
# ---

def get_fee_for_payment_method(method: str) -> dict:
    """Looks up the transaction fee percentage for a given payment method.

    This tool simulates looking up a company's internal fee structure based on
    the name of the payment method provided by the user.

    Args:
        method: The name of the payment method. It should be descriptive,
                e.g., "platinum credit card" or "bank transfer".

    Returns:
        Dictionary with status and fee information.
        Success: {"status": "success", "fee_percentage": 0.02}
        Error: {"status": "error", "error_message": "Payment method not found"}
    """
    # This simulates looking up a company's internal fee structure.
    fee_database = {
        "platinum credit card": 0.02,  # 2%
        "gold debit card": 0.035,  # 3.5%
        "bank transfer": 0.01,  # 1%
    }

    fee = fee_database.get(method.lower())
    if fee is not None:
        return {"status": "success", "fee_percentage": fee}
    else:
        return {
            "status": "error",
            "error_message": f"Payment method '{method}' not found",
        }


def get_exchange_rate(base_currency: str, target_currency: str) -> dict:
    """Looks up and returns the exchange rate between two currencies.

    Args:
        base_currency: The ISO 4217 currency code of the currency you
                       are converting from (e.g., "USD").
        target_currency: The ISO 4217 currency code of the currency you
                         are converting to (e.g., "EUR").

    Returns:
        Dictionary with status and rate information.
        Success: {"status": "success", "rate": 0.93}
        Error: {"status": "error", "error_message": "Unsupported currency pair"}
    """
    # Static data simulating a live exchange rate API
    rate_database = {
        "usd": {
            "eur": 0.93,  # Euro
            "jpy": 157.50,  # Japanese Yen
            "inr": 83.58,  # Indian Rupee
        }
    }

    # Input validation and processing
    base = base_currency.lower()
    target = target_currency.lower()

    # Return structured result with status
    rate = rate_database.get(base, {}).get(target)
    if rate is not None:
        return {"status": "success", "rate": rate}
    else:
        return {
            "status": "error",
            "error_message": f"Unsupported currency pair: {base_currency}/{target_currency}",
        }

# ---
# 3. Agent Definitions
# ---

def create_calculation_agent(retry_config):
    """
    Creates a specialist agent that ONLY generates and executes Python code
    for calculations.
    """
    calculation_agent = LlmAgent(
        name="CalculationAgent",
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        instruction="""You are a specialized calculator that ONLY responds with Python code. You are forbidden from providing any text, explanations, or conversational responses.
    
        Your task is to take a request for a calculation and translate it into a single block of Python code that calculates the answer.
        
        **RULES:**
        1.  Your output MUST be ONLY a Python code block.
        2.  Do NOT write any text before or after the code block.
        3.  The Python code MUST calculate the result.
        4.  The Python code MUST print the final result to stdout.
        5.  You are PROHIBITED from performing the calculation yourself. Your only job is to generate the code that will perform the calculation.
        
        Failure to follow these rules will result in an error.
        """,
        code_executor=BuiltInCodeExecutor(),
    )
    print("✅ Calculation specialist agent created.")
    return calculation_agent


def create_enhanced_currency_agent(retry_config, calculation_agent):
    """
    Creates the main currency agent that uses function tools AND
    the calculation_agent as a tool.
    """
    enhanced_currency_agent = LlmAgent(
        name="enhanced_currency_agent",
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        instruction="""You are a smart currency conversion assistant. You must strictly follow these steps and use the available tools.

        For any currency conversion request:

        1. Get Transaction Fee: Use the get_fee_for_payment_method() tool to determine the transaction fee.
        2. Get Exchange Rate: Use the get_exchange_rate() tool to get the currency conversion rate.
        3. Error Check: After each tool call, you must check the "status" field in the response. If the status is "error", you must stop and clearly explain the issue to the user.
        4. Calculate Final Amount (CRITICAL): You are strictly prohibited from performing any arithmetic calculations yourself. You must use the calculation_agent tool to generate Python code that calculates the final converted amount. This
           code will use the fee information from step 1 and the exchange rate from step 2.
        5. Provide Detailed Breakdown: In your summary, you must:
           * State the final converted amount.
           * Explain how the result was calculated, including:
             * The fee percentage and the fee amount in the original currency.
             * The amount remaining after deducting the fee.
             * The exchange rate applied.
        """,
        tools=[
            get_fee_for_payment_method,
            get_exchange_rate,
            AgentTool(agent=calculation_agent),  # Using the other agent as a tool!
        ],
    )
    print("✅ Enhanced currency agent created.")
    print("🔧 Available tools:")
    print("  • get_fee_for_payment_method")
    print("  • get_exchange_rate")
    print("  • CalculationAgent (as a tool)")
    return enhanced_currency_agent

# ---
# 4. Runner and Helper Functions
# ---

def create_runner(agent):
    """Create a runner to orchestrate the agent"""
    runner = InMemoryRunner(agent=agent)
    print("✅ Runner created successfully.")
    return runner


def show_python_code_and_result(response):
    """Helper function to find and print executed code from the response"""
    print("\n--- 📝 Code Execution Details ---")
    found_code = False
    for i in range(len(response)):
        # Check if the response contains a valid function call result
        if (
            (response[i].content.parts)
            and (response[i].content.parts[0])
            and (response[i].content.parts[0].function_response)
            and (response[i].content.parts[0].function_response.response)
        ):
            response_code = response[i].content.parts[0].function_response.response
            if "result" in response_code and response_code["result"] != "```":
                if "tool_code" in response_code["result"]:
                    print(
                        "Generated Python Code >> ",
                        response_code["result"].replace("tool_code", ""),
                    )
                    found_code = True
                else:
                    print("Generated Python Response >> ", response_code["result"])
                    found_code = True
    if not found_code:
        print("No Python code was executed in this turn.")
    print("----------------------------------\n")


async def run_agent_query(runner, query):
    """Run a query through the agent and display results"""
    print(f"\n{'='*80}")
    print(f"🤔 Query: {query}")
    print(f"{'='*80}\n")
    
    # .run_debug() prints the agent's thoughts (LLM turns)
    response = await runner.run_debug(query)
    
    print(f"\n{'='*80}")
    print("✅ Query completed!")
    
    # Call the helper to show any generated code
    show_python_code_and_result(response)
    
    print(f"{'='*80}\n")
    return response


async def interactive_mode(runner):
    """Run the agent in interactive mode"""
    print("\n" + "="*80)
    print("🎯 INTERACTIVE MODE - ENHANCED CURRENCY AGENT")
    print("="*80)
    print("Ask your agent questions! Type 'exit', 'quit', or 'q' to stop.")
    print("Example: Convert 1250 USD to INR using a Bank Transfer")
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
        "I want to convert 500 US Dollars to Euros using my Platinum Credit Card. How much will I receive?",
        "Convert 1,250 USD to INR using a Bank Transfer. Show me the precise calculation.",
        "How about 200 USD to JPY with a gold debit card?",
        "What happens if I use a 'normal credit card'?"
    ]
    
    print("\n" + "="*80)
    print("🚀 RUNNING PREDEFINED QUERIES")
    print("="*80 + "\n")
    
    for query in queries:
        await run_agent_query(runner, query)
        print("\n" + "-"*80 + "\n")
        await asyncio.sleep(1)  # Brief pause between queries

# ---
# 5. Main Execution
# ---

async def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("🤖 AI AGENT WITH CUSTOM TOOLS (Kaggle Day 2)")
    print("="*80 + "\n")
    
    try:
        # Setup
        setup_environment()
        retry_config = create_retry_config()
        
        # Create the specialist agent first
        calc_agent = create_calculation_agent(retry_config)
        
        # Create the main agent and pass the specialist to it
        agent = create_enhanced_currency_agent(retry_config, calc_agent)
        
        # Create the runner
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
    # Test the custom tools once before starting
    print("--- Initializing Tools ---")
    print(f"💳 Test Fee Tool: {get_fee_for_payment_method('platinum credit card')}")
    print(f"💱 Test Rate Tool: {get_exchange_rate('USD', 'EUR')}")
    print("--------------------------")
    
    # Run the async main function
    asyncio.run(main())