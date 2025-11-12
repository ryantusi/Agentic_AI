"""
Multi-Agent Systems & Workflow Patterns
A comprehensive script demonstrating all multi-agent patterns using Google's ADK
"""

import os
import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
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
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504]
    )
    print("✅ Retry configuration created.")
    return retry_config


# ============================================================================
# PATTERN 1: LLM-BASED MULTI-AGENT (Dynamic Orchestration)
# ============================================================================

def create_research_summarizer_system(retry_config):
    """
    Pattern: LLM-based multi-agent coordination
    Use case: When you need dynamic orchestration and the LLM should decide the order
    
    Creates a research and summarization system where a coordinator agent
    dynamically orchestrates research and summarization agents.
    """
    print("\n" + "="*80)
    print("🔧 CREATING LLM-BASED MULTI-AGENT SYSTEM")
    print("="*80)
    
    # Research Agent: Uses Google Search to find information
    research_agent = Agent(
        name="ResearchAgent",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="""You are a specialized research agent. Your only job is to use the
        google_search tool to find 2-3 pieces of relevant information on the given topic 
        and present the findings with citations.""",
        tools=[google_search],
        output_key="research_findings",
    )
    
    # Summarizer Agent: Creates concise summaries
    summarizer_agent = Agent(
        name="SummarizerAgent",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="""You are a specialized summarizer. Your job is to take research findings 
        and create a concise, well-structured summary. Research findings: {research_findings}
        Create a summary that is clear, informative, and around 100-150 words.""",
        output_key="summary",
    )
    
    # Root Coordinator: Orchestrates the workflow
    root_agent = Agent(
        name="ResearchCoordinator",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="""You are a research coordinator. Your goal is to answer the user's query 
        by orchestrating a workflow.
        1. First, you MUST call the `ResearchAgent` tool to find relevant information.
        2. Next, after receiving the research findings, you MUST call the `SummarizerAgent` 
           tool to create a concise summary.
        3. Finally, present the final summary clearly to the user as your response.""",
        tools=[AgentTool(research_agent), AgentTool(summarizer_agent)],
    )
    
    print("✅ LLM-based multi-agent system created.")
    return root_agent


# ============================================================================
# PATTERN 2: SEQUENTIAL WORKFLOWS (The Assembly Line)
# ============================================================================

def create_blog_pipeline(retry_config):
    """
    Pattern: Sequential workflow
    Use case: When order matters and each step builds on the previous one
    
    Creates a blog post creation pipeline: Outline → Write → Edit
    """
    print("\n" + "="*80)
    print("🔧 CREATING SEQUENTIAL WORKFLOW SYSTEM")
    print("="*80)
    
    # Outline Agent: Creates the initial blog post outline
    outline_agent = Agent(
        name="OutlineAgent",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="""Create a blog outline for the given topic with:
        1. A catchy headline
        2. An introduction hook
        3. 3-5 main sections with 2-3 bullet points for each
        4. A concluding thought""",
        output_key="blog_outline",
    )
    
    # Writer Agent: Writes the full blog post
    writer_agent = Agent(
        name="WriterAgent",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="""Following this outline strictly: {blog_outline}
        Write a brief, 200 to 300-word blog post with an engaging and informative tone.""",
        output_key="blog_draft",
    )
    
    # Editor Agent: Edits and polishes the draft
    editor_agent = Agent(
        name="EditorAgent",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="""Edit this draft: {blog_draft}
        Your task is to polish the text by fixing any grammatical errors, improving the 
        flow and sentence structure, and enhancing overall clarity.""",
        output_key="final_blog",
    )
    
    # Sequential Agent: Runs agents in order
    root_agent = SequentialAgent(
        name="BlogPipeline",
        sub_agents=[outline_agent, writer_agent, editor_agent],
    )
    
    print("✅ Sequential workflow system created.")
    return root_agent


# ============================================================================
# PATTERN 3: PARALLEL WORKFLOWS (Independent Researchers)
# ============================================================================

def create_parallel_research_system(retry_config):
    """
    Pattern: Parallel workflow
    Use case: When tasks are independent and speed matters
    
    Creates a multi-topic research system that researches tech, health, and 
    finance topics concurrently, then aggregates the results.
    """
    print("\n" + "="*80)
    print("🔧 CREATING PARALLEL WORKFLOW SYSTEM")
    print("="*80)
    
    # Tech Researcher
    tech_researcher = Agent(
        name="TechResearcher",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="""Research the latest AI/ML trends. Include 3 key developments,
        the main companies involved, and the potential impact. Keep the report very 
        concise (100 words).""",
        tools=[google_search],
        output_key="tech_research",
    )
    
    # Health Researcher
    health_researcher = Agent(
        name="HealthResearcher",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="""Research recent medical breakthroughs. Include 3 significant advances,
        their practical applications, and estimated timelines. Keep the report concise 
        (100 words).""",
        tools=[google_search],
        output_key="health_research",
    )
    
    # Finance Researcher
    finance_researcher = Agent(
        name="FinanceResearcher",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="""Research current fintech trends. Include 3 key trends,
        their market implications, and the future outlook. Keep the report concise 
        (100 words).""",
        tools=[google_search],
        output_key="finance_research",
    )
    
    # Aggregator Agent: Combines all research findings
    aggregator_agent = Agent(
        name="AggregatorAgent",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="""Combine these three research findings into a single executive summary:
        **Technology Trends:**
        {tech_research}
        
        **Health Breakthroughs:**
        {health_research}
        
        **Finance Innovations:**
        {finance_research}
        
        Your summary should highlight common themes, surprising connections, and the most 
        important key takeaways from all three reports. The final summary should be around 
        200 words.""",
        output_key="executive_summary",
    )
    
    # Parallel Agent: Runs all researchers simultaneously
    parallel_research_team = ParallelAgent(
        name="ParallelResearchTeam",
        sub_agents=[tech_researcher, health_researcher, finance_researcher],
    )
    
    # Sequential Agent: Run parallel team first, then aggregator
    root_agent = SequentialAgent(
        name="ResearchSystem",
        sub_agents=[parallel_research_team, aggregator_agent],
    )
    
    print("✅ Parallel workflow system created.")
    return root_agent


# ============================================================================
# PATTERN 4: LOOP WORKFLOWS (The Refinement Cycle)
# ============================================================================

def create_story_refinement_system(retry_config):
    """
    Pattern: Loop workflow
    Use case: When iterative improvement and quality refinement is needed
    
    Creates a story writing and critique loop where a writer creates drafts
    and a critic provides feedback until the story is approved.
    """
    print("\n" + "="*80)
    print("🔧 CREATING LOOP WORKFLOW SYSTEM")
    print("="*80)
    
    # Initial Writer Agent: Creates the first draft
    initial_writer_agent = Agent(
        name="InitialWriterAgent",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="""Based on the user's prompt, write the first draft of a short story 
        (around 100-150 words). Output only the story text, with no introduction or 
        explanation.""",
        output_key="current_story",
    )
    
    # Critic Agent: Reviews and critiques the story
    critic_agent = Agent(
        name="CriticAgent",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="""You are a constructive story critic. Review the story provided below.
        Story: {current_story}
        
        Evaluate the story's plot, characters, and pacing.
        - If the story is well-written and complete, you MUST respond with the exact phrase: "APPROVED"
        - Otherwise, provide 2-3 specific, actionable suggestions for improvement.""",
        output_key="critique",
    )
    
    # Exit loop function
    def exit_loop():
        """Call this function ONLY when the critique is 'APPROVED'"""
        return {"status": "approved", "message": "Story approved. Exiting refinement loop."}
    
    # Refiner Agent: Refines the story based on critique
    refiner_agent = Agent(
        name="RefinerAgent",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="""You are a story refiner. You have a story draft and critique.
        
        Story Draft: {current_story}
        Critique: {critique}
        
        Your task is to analyze the critique.
        - IF the critique is EXACTLY "APPROVED", you MUST call the `exit_loop` function.
        - OTHERWISE, rewrite the story draft to fully incorporate the feedback.""",
        output_key="current_story",
        tools=[FunctionTool(exit_loop)],
    )
    
    # Loop Agent: Runs critic and refiner repeatedly
    story_refinement_loop = LoopAgent(
        name="StoryRefinementLoop",
        sub_agents=[critic_agent, refiner_agent],
        max_iterations=2,
    )
    
    # Sequential Agent: Initial write → Refinement loop
    root_agent = SequentialAgent(
        name="StoryPipeline",
        sub_agents=[initial_writer_agent, story_refinement_loop],
    )
    
    print("✅ Loop workflow system created.")
    return root_agent


# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

async def run_workflow(agent, query, workflow_name):
    """Run a workflow and display results"""
    print("\n" + "="*80)
    print(f"🚀 RUNNING: {workflow_name}")
    print("="*80)
    print(f"📝 Query: {query}")
    print("="*80 + "\n")
    
    runner = InMemoryRunner(agent=agent)
    
    try:
        response = await runner.run_debug(query)
        print(f"\n{'='*80}")
        print(f"✅ {workflow_name} COMPLETED!")
        print(f"{'='*80}\n")
        return response
    except Exception as e:
        print(f"\n❌ Error in {workflow_name}: {e}\n")
        raise


async def run_all_patterns(retry_config):
    """Run all multi-agent patterns with example queries"""
    
    print("\n" + "="*80)
    print("🎯 RUNNING ALL MULTI-AGENT PATTERNS")
    print("="*80)
    
    # Pattern 1: LLM-based Multi-Agent
    print("\n📌 PATTERN 1: LLM-BASED MULTI-AGENT (Dynamic Orchestration)")
    research_system = create_research_summarizer_system(retry_config)
    await run_workflow(
        research_system,
        "What are the latest advancements in quantum computing and what do they mean for AI?",
        "LLM-Based Multi-Agent System"
    )
    await asyncio.sleep(3)
    
    # Pattern 2: Sequential Workflow
    print("\n📌 PATTERN 2: SEQUENTIAL WORKFLOW (Assembly Line)")
    blog_pipeline = create_blog_pipeline(retry_config)
    await run_workflow(
        blog_pipeline,
        "Write a blog post about the benefits of multi-agent systems for software developers",
        "Sequential Blog Pipeline"
    )
    await asyncio.sleep(3)
    
    # Pattern 3: Parallel Workflow
    print("\n📌 PATTERN 3: PARALLEL WORKFLOW (Concurrent Execution)")
    parallel_system = create_parallel_research_system(retry_config)
    await run_workflow(
        parallel_system,
        "Run the daily executive briefing on Tech, Health, and Finance",
        "Parallel Research System"
    )
    await asyncio.sleep(3)
    
    # Pattern 4: Loop Workflow
    print("\n📌 PATTERN 4: LOOP WORKFLOW (Iterative Refinement)")
    story_system = create_story_refinement_system(retry_config)
    await run_workflow(
        story_system,
        "Write a short story about a lighthouse keeper who discovers a mysterious, glowing map",
        "Loop Story Refinement System"
    )


async def interactive_mode(retry_config):
    """Interactive mode to choose and run specific patterns"""
    
    print("\n" + "="*80)
    print("🎮 INTERACTIVE MODE - CHOOSE A PATTERN")
    print("="*80)
    
    while True:
        print("\nAvailable Multi-Agent Patterns:")
        print("1. LLM-Based Multi-Agent (Research + Summarize)")
        print("2. Sequential Workflow (Blog: Outline → Write → Edit)")
        print("3. Parallel Workflow (Multi-Topic Research)")
        print("4. Loop Workflow (Story Writing with Refinement)")
        print("5. Run All Patterns")
        print("6. Exit")
        
        choice = input("\nSelect a pattern (1-6): ").strip()
        
        if choice == "6":
            print("\n👋 Goodbye!")
            break
        
        if choice == "1":
            agent = create_research_summarizer_system(retry_config)
            query = input("\nEnter your research topic: ").strip()
            if query:
                await run_workflow(agent, query, "LLM-Based Multi-Agent")
        
        elif choice == "2":
            agent = create_blog_pipeline(retry_config)
            query = input("\nEnter your blog topic: ").strip()
            if query:
                await run_workflow(agent, query, "Sequential Blog Pipeline")
        
        elif choice == "3":
            agent = create_parallel_research_system(retry_config)
            await run_workflow(
                agent,
                "Run the daily executive briefing on Tech, Health, and Finance",
                "Parallel Research System"
            )
        
        elif choice == "4":
            agent = create_story_refinement_system(retry_config)
            query = input("\nEnter your story prompt: ").strip()
            if query:
                await run_workflow(agent, query, "Loop Story Refinement")
        
        elif choice == "5":
            await run_all_patterns(retry_config)
        
        else:
            print("❌ Invalid choice. Please select 1-6.")
        
        print("\n" + "-"*80)


async def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("🤖 MULTI-AGENT SYSTEMS & WORKFLOW PATTERNS")
    print("="*80 + "\n")
    
    try:
        # Setup
        setup_environment()
        retry_config = create_retry_config()
        
        print("\n" + "="*80)
        print("✅ ALL COMPONENTS INITIALIZED SUCCESSFULLY!")
        print("="*80)
        
        # Choose mode
        print("\nSelect mode:")
        print("1. Run all patterns with example queries")
        print("2. Interactive mode (choose specific patterns)")
        
        choice = input("\nEnter your choice (1/2): ").strip()
        
        if choice == "1":
            await run_all_patterns(retry_config)
        elif choice == "2":
            await interactive_mode(retry_config)
        else:
            print("Invalid choice. Running all patterns...")
            await run_all_patterns(retry_config)
        
        print("\n" + "="*80)
        print("🎉 ALL WORKFLOWS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n📚 Pattern Summary:")
        print("  • LLM-Based: Dynamic orchestration by LLM")
        print("  • Sequential: Deterministic order, assembly line")
        print("  • Parallel: Concurrent execution for speed")
        print("  • Loop: Iterative refinement cycles")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}\n")
        raise


if __name__ == "__main__":
    asyncio.run(main())
