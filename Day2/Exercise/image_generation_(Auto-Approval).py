"""
🎯 Day 2 Exercise: Image Generation with Cost Approval (Gatekeeper Pattern)

Scenario:
- Single Image (1): Auto-generate (Fast)
- Bulk Images (>1): Pause for Human Approval (Safety/Cost Control)

Tech Stack:
- MCP Server: @modelcontextprotocol/server-everything (providing 'getTinyImage')
- ADK Pattern: Long-Running Operation (LRO) with ToolContext
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

# A. The MCP Tool (The "Expensive" Generator)
def create_mcp_toolset():
    """Connects to the public 'Everything' MCP server."""
    cmd = "npx.cmd" if sys.platform == "win32" else "npx"
    try:
        return McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=cmd, 
                    args=["-y", "@modelcontextprotocol/server-everything"],
                    tool_filter=["getTinyImage"], # We only want this tool
                ),
                timeout=30,
            )
        )
    except Exception as e:
        print(f"❌ MCP Error: {e}")
        return None

# B. The Gatekeeper Tool (Custom Logic)
def validate_image_batch(prompt: str, count: int, tool_context: ToolContext) -> dict:
    """
    Validates an image generation request.
    Acts as a gatekeeper for cost control.
    
    Args:
        prompt: Description of image
        count: Number of images requested
    """
    print(f"   🛡️ [Gatekeeper] Validating request for {count} image(s)...")

    # Rule 1: Single images are free/safe -> Auto-approve
    if count <= 1:
        return {
            "status": "APPROVED", 
            "message": "Single image request auto-approved. Proceed to generate."
        }

    # Rule 2: Bulk images cost money -> Pause for Approval
    # 2a. First time call (Pause)
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"Bulk Generation Request: {count} images for '{prompt}'. Approve cost?",
            payload={"count": count, "prompt": prompt}
        )
        return {"status": "PENDING", "message": "Approval required for bulk generation."}

    # 2b. Resumed call (Check decision)
    if tool_context.tool_confirmation.confirmed:
        return {
            "status": "APPROVED", 
            "message": f"Bulk request for {count} images APPROVED by human admin."
        }
    else:
        return {
            "status": "DENIED", 
            "message": "Bulk request rejected. Do not generate images."
        }

# --- 3. The Workflow Engine ---

# Helper to find the pause signal
def check_for_pause(events):
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if (part.function_call and 
                    part.function_call.name == "adk_request_confirmation"):
                    return {"id": part.function_call.id, "invocation_id": event.invocation_id}
    return None

# Helper to format the human's decision
def create_human_decision(approval_info, is_approved: bool):
    return types.Content(
        role="user", 
        parts=[types.Part(function_response=types.FunctionResponse(
            id=approval_info["id"],
            name="adk_request_confirmation",
            response={"confirmed": is_approved},
        ))]
    )

async def run_agent_workflow(runner, session_service, query, auto_approve=True):
    print(f"\n{'='*50}\nUser > {query}\n{'='*50}")
    
    # Create a fresh session
    session_id = f"sess_{uuid.uuid4().hex[:6]}"
    
    # FIX: Use Explicit Keyword Arguments here
    await session_service.create_session(
        app_name="image_app", 
        user_id="user1", 
        session_id=session_id
    )
    
    # Step 1: Initial Run
    events = []
    print("▶️  Step 1: Agent thinking...")
    
    # Ensure explicit arguments here as well for safety
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

    # Step 2: Check for Pause (Gatekeeper triggered)
    pause_info = check_for_pause(events)
    
    if pause_info:
        decision_str = "✅ APPROVE" if auto_approve else "❌ REJECT"
        print(f"\n⏸️  GATEKEEPER PAUSED: Bulk request detected.")
        print(f"👤 Admin Decision: {decision_str}")
        
        print("\n▶️  Step 2: Resuming Agent...")
        async for event in runner.run_async(
            user_id="user1", 
            session_id=session_id, 
            new_message=create_human_decision(pause_info, auto_approve),
            invocation_id=pause_info["invocation_id"]
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text: print(f"🤖 {part.text}")
                    if part.function_call and part.function_call.name == "getTinyImage":
                         print("   🎨 [MCP] Generating Image...")
    else:
        print("\n✅ Finished without pause.")

# --- 4. Main Execution ---

async def main_logic():
    print("\n🚀 EXERCISE SOLUTION: IMAGE GEN WITH APPROVAL")
    setup_environment()
    
    # 1. Setup Tools
    mcp_tool = create_mcp_toolset()
    if not mcp_tool: return # Exit if Node.js missing

    gatekeeper_tool = FunctionTool(func=validate_image_batch)
    
    # 2. Setup Agent
    # CRITICAL: We give strict instructions on tool order
    agent = LlmAgent(
        name="creative_agent",
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=create_retry_config()),
        instruction="""
        You are an Image Generation Assistant.
        
        PROTOCOL:
        1. Whenever a user asks for images, you MUST first call `validate_image_batch` to check cost rules.
        2. IF validation returns 'APPROVED':
           - Use the `getTinyImage` tool from the MCP server.
           - If multiple images are requested, call `getTinyImage` multiple times (or as appropriate).
        3. IF validation returns 'DENIED':
           - Apologize and do not generate any images.
        """,
        tools=[gatekeeper_tool, mcp_tool]
    )

    # 3. Setup App (Resumable)
    app = App(name="image_app", root_agent=agent, resumability_config=ResumabilityConfig(is_resumable=True))
    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service)

    try:
        while True:
            print("\n" + "-"*30)
            print("1. Test: Single Image (Should Auto-Approve)")
            print("2. Test: Bulk Images (Should Pause -> Approve)")
            print("3. Test: Bulk Images (Should Pause -> Reject)")
            print("q. Quit")
            choice = input("Select: ").strip().lower()

            if choice == '1': 
                await run_agent_workflow(runner, session_service, "Generate 1 tiny image of a cat")
            elif choice == '2': 
                await run_agent_workflow(runner, session_service, "Generate 3 tiny images of space", auto_approve=True)
            elif choice == '3': 
                await run_agent_workflow(runner, session_service, "Generate 50 tiny images of clouds", auto_approve=False)
            elif choice == 'q': 
                break
    finally:
        print("\n🧹 Cleaning up...")
        if hasattr(runner, 'close'): await runner.close()

if __name__ == "__main__":
    asyncio.run(main_logic())