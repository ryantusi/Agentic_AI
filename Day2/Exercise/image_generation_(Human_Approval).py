"""
🎯 Day 2 Exercise: Interactive Image Generation Agent

Changes:
- Real Human-in-the-Loop: The script now PAUSES and waits for your keyboard input.
- 'Gatekeeper' Pattern: Validates costs before generating.
"""

import os
import uuid
import asyncio
import sys
from dotenv import load_dotenv

# Google ADK Imports
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext

# MCP Imports
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

# --- 1. Setup ---

def setup_environment():
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  WARNING: GOOGLE_API_KEY not found.")

def create_retry_config():
    return types.HttpRetryOptions(
        attempts=3, exp_base=2, initial_delay=1,
        http_status_codes=[429, 500, 503, 504]
    )

# --- 2. The Tools ---

def create_mcp_toolset():
    cmd = "npx.cmd" if sys.platform == "win32" else "npx"
    try:
        return McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=cmd, 
                    args=["-y", "@modelcontextprotocol/server-everything"],
                    tool_filter=["getTinyImage"], 
                ),
                timeout=30,
            )
        )
    except Exception as e:
        print(f"❌ MCP Error: {e}")
        return None

def validate_image_batch(prompt: str, count: int, tool_context: ToolContext) -> dict:
    """
    Gatekeeper Tool: Pauses for confirmation if count > 1.
    """
    print(f"   🛡️ [Gatekeeper] Checking request for {count} image(s)...")

    # Rule 1: Single images are free -> Auto-approve
    if count <= 1:
        return {"status": "APPROVED", "message": "Single image auto-approved."}

    # Rule 2: Bulk images -> Pause
    if not tool_context.tool_confirmation:
        # This hint is what the system 'sees' internally
        tool_context.request_confirmation(
            hint=f"Approve bulk generation of {count} images?",
            payload={"count": count, "prompt": prompt}
        )
        return {"status": "PENDING", "message": "Waiting for human approval..."}

    # Rule 3: Resume with decision
    if tool_context.tool_confirmation.confirmed:
        return {"status": "APPROVED", "message": "Bulk request APPROVED by admin."}
    else:
        return {"status": "DENIED", "message": "Bulk request DENIED by admin."}

# --- 3. The Workflow Engine (With REAL Input) ---

def check_for_pause(events):
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if (part.function_call and 
                    part.function_call.name == "adk_request_confirmation"):
                    return {"id": part.function_call.id, "invocation_id": event.invocation_id}
    return None

def create_human_decision(approval_info, is_approved: bool):
    return types.Content(
        role="user", 
        parts=[types.Part(function_response=types.FunctionResponse(
            id=approval_info["id"],
            name="adk_request_confirmation",
            response={"confirmed": is_approved},
        ))]
    )

async def run_interactive_workflow(runner, session_service, query):
    print(f"\n{'='*50}\nUser > {query}\n{'='*50}")
    
    session_id = f"sess_{uuid.uuid4().hex[:6]}"
    await session_service.create_session(
        app_name="image_app", 
        user_id="user1", 
        session_id=session_id
    )
    
    # --- Phase 1: Initial Request ---
    events = []
    print("▶️  Step 1: Agent processing...")
    async for event in runner.run_async(
        user_id="user1", 
        session_id=session_id, 
        new_message=types.Content(role="user", parts=[types.Part(text=query)])
    ):
        events.append(event)
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text: print(f"🤖 {part.text}")
                if part.function_call and part.function_call.name == "getTinyImage":
                     print("   🎨 [MCP] Generating Image...")

    # --- Phase 2: Check for Pause ---
    pause_info = check_for_pause(events)
    
    if pause_info:
        print(f"\n⏸️  GATEKEEPER PAUSED: Bulk request detected.")
        
        # --- INTERACTIVE INPUT ---
        # This is where the script actually stops and waits for you!
        while True:
            decision = input("👤 Admin: Do you approve this cost? (y/n): ").strip().lower()
            if decision in ['y', 'yes']:
                is_approved = True
                print("   -> Decision: APPROVED ✅")
                break
            elif decision in ['n', 'no']:
                is_approved = False
                print("   -> Decision: REJECTED ❌")
                break
            else:
                print("   Please type 'y' or 'n'.")

        # --- Phase 3: Resume with YOUR decision ---
        print("\n▶️  Step 2: Resuming Agent...")
        async for event in runner.run_async(
            user_id="user1", 
            session_id=session_id, 
            new_message=create_human_decision(pause_info, is_approved),
            invocation_id=pause_info["invocation_id"]
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text: print(f"🤖 {part.text}")
                    if part.function_call and part.function_call.name == "getTinyImage":
                         print("   🎨 [MCP] Generating Image...")
    else:
        print("\n✅ Request completed automatically.")

# --- 4. Main Execution ---

async def main_logic():
    print("\n🚀 EXERCISE: INTERACTIVE GATEKEEPER")
    setup_environment()
    
    mcp_tool = create_mcp_toolset()
    if not mcp_tool: return 

    agent = LlmAgent(
        name="creative_agent",
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=create_retry_config()),
        instruction="""
        You are an Image Generation Assistant.
        PROTOCOL:
        1. ALWAYS call `validate_image_batch` first.
        2. IF 'APPROVED': Call `getTinyImage` (loop if needed for multiple).
        3. IF 'DENIED': Do not generate. Apologize.
        """,
        tools=[FunctionTool(func=validate_image_batch), mcp_tool]
    )

    app = App(name="image_app", root_agent=agent, resumability_config=ResumabilityConfig(is_resumable=True))
    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service)

    try:
        while True:
            print("\n" + "-"*30)
            print("Type a request (or 'q' to quit)")
            print("Examples:")
            print(" - 'Generate 1 tiny image of a cat'")
            print(" - 'Generate 5 tiny images of space'")
            
            user_input = input("\nYour Request > ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            
            if user_input:
                await run_interactive_workflow(runner, session_service, user_input)

    finally:
        print("\n🧹 Cleaning up...")
        if hasattr(runner, 'close'): await runner.close()

if __name__ == "__main__":
    asyncio.run(main_logic())