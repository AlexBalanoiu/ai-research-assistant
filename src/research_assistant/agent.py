"""
Research agent: reasoning + calculator + web search + persistent memory
of past research, with query refinement, source citation, tool-call
budget awareness, and a structured final answer.
"""
import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from research_assistant.tools.calculator import calculator
from research_assistant.tools.web_search import web_search
from research_assistant.tools.past_research import check_past_research

load_dotenv()


def build_agent(model_id: str | None = None) -> Agent:
    resolved_model_id = model_id or os.environ.get("MODEL_ID", "ollama_chat/llama3.1")

    return Agent(
        model=LiteLlm(model=resolved_model_id),
        name="research_assistant_v1",
        description="Research agent with tools, persistent memory, query refinement, citations, and budget",
        instruction=(
            "You are a research assistant with three tools:\n"
            "- calculator: ALWAYS use this for any arithmetic or math "
            "expression, no matter how simple (even 2+2). Never compute "
            "math yourself, even mentally - always call the tool.\n"
            "- check_past_research: for any research question (not math, "
            "not a trivial known fact), call this FIRST with a short "
            "topic keyword. If it returns relevant past findings, reuse "
            "them (mention they come from earlier research) instead of "
            "searching again. If nothing relevant comes back, proceed "
            "with web_search normally.\n"
            "- web_search: use for current events, facts about specific "
            "entities, prices, or anything you cannot be certain about "
            "from reasoning alone, after checking past research first.\n"
            "Do not use a tool when you can answer directly and reliably "
            "from general knowledge (non-math facts). Never use web_search "
            "for arithmetic.\n\n"
            "Query refinement: each web_search result includes a 'count' "
            "field. If count is 0 or the results look irrelevant, "
            "reformulate the query with different keywords and search "
            "again (at most 2 extra attempts) before telling the user you "
            "could not find reliable information.\n\n"
            "Tool budget: there is a hard limit on tool calls per "
            "question. If a tool returns a budget-exceeded error, stop "
            "calling tools and answer with whatever you already have, "
            "noting what could not be verified.\n\n"
            "For any question that required research (web_search or "
            "reused past research), end your final response with exactly "
            "this structure:\n\n"
            "### Synthesis\n"
            "<your full reasoned answer. When you use a fact from a "
            "search result, mention the source by name inline, e.g. "
            "\"According to Wikipedia...\">\n\n"
            "### Conclusion\n"
            "<a one or two sentence bottom-line takeaway>\n\n"
            "For simple questions that did not need research (math, "
            "trivial facts), just answer directly without these markers."
        ),
        tools=[calculator, web_search, check_past_research],
    )


root_agent = build_agent()