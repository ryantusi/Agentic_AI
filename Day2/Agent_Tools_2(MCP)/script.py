"""
🚀 Agent Tool Patterns (Kaggle Day 2)

A script to demonstrate advanced agent patterns using Google ADK:
1. MCP (Model Context Protocol): Connecting to external tools (Node.js server).
2. Long-Running Operations (LRO): Handling Human-in-the-Loop workflows (Pause/Resume).

Prerequisites:
- pip install google-agent-development-kit python-dotenv mcp
- Node.js installed (for the 'npx' command used in the MCP section)
"""

import os
import uuid
import asyncio
import base64
from dotenv import load_dotenv

# Google ADK Imports
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool

# MCP Specific Imports
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

# Tool Context for LRO
from google.adk.tools.tool_context import ToolContext

# ---
# 1. Environment and Configuration
# ---

def setup_environment():
    """Load environment variables."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("🔑 GOOGLE_API_KEY not found. Check your .env file.")
    os.environ["GOOGLE_API_KEY"] = api_key
    print("✅ Environment setup complete.")

def create_retry_config():
    return types.HttpRetryOptions(
        attempts=5, exp_base=7, initial_delay=1,
        http_status_codes=[429, 500, 503, 504]
    )

# ---
# 2. PART A: Model Context Protocol (MCP)
# ---

def create_mcp_toolset():
    """
    Creates a toolset that connects to the 'Everything' MCP server via npx.
    Requires Node.js installed on the system.
    """
    print("\n⏳ Initializing MCP Toolset (connecting to @modelcontextprotocol/server-everything)...")
    try:
        mcp_server = McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="npx",  # This requires Node.js/npx in your PATH
                    args=[
                        "-y",
                        "@modelcontextprotocol/server-everything",
                    ],
                    tool_filter=["getTinyImage"], # Only expose this specific tool
                ),
                timeout=60, # Giving npx time to install if needed
            )
        )
        print("✅ MCP Toolset connected successfully.")
        return mcp_server
    except Exception as e:
        print(f"❌ MCP Connection Failed: {e}")
        print("👉 Ensure you have Node.js installed and 'npx' is in your system PATH.")
        return None

async def run_mcp_demo(retry_config):
    """Runs the MCP demo: Generating a tiny image."""
    print("\n" + "="*60)
    print("🖼️  PART A: MCP DEMO (Tiny Image)")
    print("="*60)

    mcp_tool = create_mcp_toolset()
    if not mcp_tool:
        return

    agent = LlmAgent(
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        name="image_agent",
        instruction="Use the MCP Tool to generate images for user queries.",
        tools=[mcp_tool],
    )

    runner = InMemoryRunner(agent=agent)
    
    query = "Provide a sample tiny image"
    print(f"\nUser > {query}")
    
    response = await runner.run_debug(query, verbose=True)
    
    # Helper to inspect the complex response structure for image data
    print("\n--- 🔍 Inspecting MCP Response for Images ---")
    for event in response:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_response") and part.function_response:
                    response_data = part.function_response.response
                    for item in response_data.get("content", []):
                        if item.get("type") == "image":
                            print(f"✅ Image received! Format: {item.get('mimeType')}")
                            print(f"📦 Base64 Data (truncated): {item.get('data')[:50]}...")
                            # In a notebook, we would display(IPImage(...)), 
                            # here we just confirm receipt.

# ---
# 3. PART B: Long-Running Operations (Human-in-the-Loop)
# ---

LARGE_ORDER_THRESHOLD = 5

def place_shipping_order(
    num_containers: int, destination: str, tool_context: ToolContext
) -> dict:
    """
    Places a shipping order. Requires approval if ordering more than 5 containers.
    
    Args:
        num_containers: Number of containers to ship
        destination: Shipping destination
    """
    print(f"   [Tool] Processing order: {num_containers} to {destination}...")

    # SCENARIO 1: Auto-approve small orders
    if num_containers <= LARGE_ORDER_THRESHOLD:
        print("   [Tool] Small order. Auto-approving.")
        return {
            "status": "approved",
            "order_id": f"ORD-{num_containers}-AUTO",
            "message": f"Order auto-approved: {num_containers} containers to {destination}",
        }

    # SCENARIO 2: Large order - FIRST CALL (Pause)
    if not tool_context.tool_confirmation:
        print("   [Tool] ⚠️ Large order detected. Requesting confirmation...")
        tool_context.request_confirmation(
            hint=f"Large order: {num_containers} containers to {destination}. Approve?",
            payload={"num_containers": num_containers, "destination": destination},
        )
        return {
            "status": "pending",
            "message": f"Order for {num_containers} containers requires approval",
        }

    # SCENARIO 3: Large order - RESUMED CALL (Check Decision)
    if tool_context.tool_confirmation.confirmed:
        print("   [Tool] ✅ Confirmation received! Approving.")
        return {
            "status": "approved",
            "order_id": f"ORD-{num_containers}-HUMAN",
            "message": f"Order approved: {num_containers} containers to {destination}",
        }
    else:
        print("   [Tool] ❌ Confirmation denied. Rejecting.")
        return {
            "status": "rejected",
            "message": f"Order rejected: {num_containers} containers to {destination}",
        }

# --- 4. Workflow Helpers ---

def check_for_approval(events):
    """Scans events for the specific 'adk_request_confirmation' function call."""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if (part.function_call and 
                    part.function_call.name == "adk_request_confirmation"):
                    return {
                        "approval_id": part.function_call.id,
                        "invocation_id": event.invocation_id,
                    }
    return None

def create_approval_response(approval_info, approved: bool):
    """Creates the formatted response structure ADK expects for confirmation."""
    confirmation_response = types.FunctionResponse(
        id=approval_info["approval_id"],
        name="adk_request_confirmation",
        response={"confirmed": approved},
    )
    return types.Content(
        role="user", parts=[types.Part(function_response=confirmation_response)]
    )

def print_agent_text(events):
    """Extracts and prints text from agent events."""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"🤖 Agent > {part.text}")

# --- 5. The Workflow Runner ---

async def run_shipping_workflow(
    runner, session_service, query: str, auto_approve_decision: bool = True
):
    """
    Orchestrates the Pause/Resume workflow.
    """
    print(f"\n{'='*60}")
    print(f"User Query > {query}")
    print(f"{'='*60}")

    # Unique session ID ensures clean state for each run
    session_id = f"order_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name="shipping_coordinator", user_id="test_user", session_id=session_id
    )

    query_content = types.Content(role="user", parts=[types.Part(text=query)])
    events = []

    # STEP 1: Initial Run
    print("▶️  Step 1: Sending initial request...")
    async for event in runner.run_async(
        user_id="test_user", session_id=session_id, new_message=query_content
    ):
        events.append(event)

    # STEP 2: Check for Pause
    approval_info = check_for_approval(events)

    if approval_info:
        # STEP 3: Handle Pause (Human-in-the-Loop)
        print(f"\n⏸️  Workflow Paused: Agent requested approval.")
        decision_str = "APPROVE ✅" if auto_approve_decision else "REJECT ❌"
        print(f"👤 Human Decision: {decision_str}")
        
        print("\n▶️  Step 2: Resuming Agent with decision...")
        
        # RESUME call - Must pass 'invocation_id' to resume correct thread
        async for event in runner.run_async(
            user_id="test_user",
            session_id=session_id,
            new_message=create_approval_response(approval_info, auto_approve_decision),
            invocation_id=approval_info["invocation_id"], # CRITICAL!
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"🤖 Agent (Resumed) > {part.text}")
    else:
        # No pause needed
        print("\n✅ Workflow Completed without pause.")
        print_agent_text(events)

# ---
# 6. Main Execution
# ---

async def main():
    print("\n🚀 STARTING DAY 2: AGENT PATTERNS")
    
    try:
        setup_environment()
        retry_config = create_retry_config()

        # --- Setup for Shipping Agent (Part B) ---
        shipping_agent = LlmAgent(
            name="shipping_agent",
            model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
            instruction="""You are a shipping coordinator.
            1. Use place_shipping_order for all requests.
            2. If status is 'pending', inform user approval is needed.
            3. Provide a final summary with Order ID and status.""",
            tools=[FunctionTool(func=place_shipping_order)],
        )

        # Wrap in App for RESUMABILITY
        shipping_app = App(
            name="shipping_coordinator",
            root_agent=shipping_agent,
            resumability_config=ResumabilityConfig(is_resumable=True),
        )
        
        session_service = InMemorySessionService()
        shipping_runner = Runner(app=shipping_app, session_service=session_service)

        # --- Menu ---
        while True:
            print("\n" + "="*30)
            print("Select a Demo:")
            print("1. MCP Demo (Connect to Node.js Server)")
            print("2. Workflow: Small Order (Auto-Approve)")
            print("3. Workflow: Large Order (Approve)")
            print("4. Workflow: Large Order (Reject)")
            print("q. Quit")
            
            choice = input("Enter choice: ").strip().lower()

            if choice == '1':
                await run_mcp_demo(retry_config)
            elif choice == '2':
                await run_shipping_workflow(
                    shipping_runner, session_service, 
                    "Ship 3 containers to Singapore"
                )
            elif choice == '3':
                await run_shipping_workflow(
                    shipping_runner, session_service, 
                    "Ship 10 containers to Rotterdam", 
                    auto_approve_decision=True
                )
            elif choice == '4':
                await run_shipping_workflow(
                    shipping_runner, session_service, 
                    "Ship 8 containers to Los Angeles", 
                    auto_approve_decision=False
                )
            elif choice == 'q':
                print("👋 Goodbye!")
                break
            else:
                print("Invalid choice.")

    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())