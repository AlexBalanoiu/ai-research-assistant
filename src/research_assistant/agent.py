"""
Step 5 - Agent + calculator + web search + structured final answer.
"""
import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from research_assistant.tools.calculator import calculator
from research_assistant.tools.web_search import web_search

load_dotenv()


def build_agent() -> Agent:
    model_id = os.environ.get("MODEL_ID", "ollama_chat/llama3.1")

    return Agent(
        model=LiteLlm(model=model_id),
        name="research_assistant_v1",
        description="Research agent - step 5: reasoning + tools + structured report output",
        instruction=(
            "You are a research assistant with two tools:\n"
            "- calculator: ALWAYS use this for any arithmetic or math "
            "expression, no matter how simple (even 2+2). Never compute "
            "math yourself, even mentally - always call the tool.\n"
            "- web_search: use for current events, facts about specific "
            "entities, prices, or anything you cannot be certain about "
            "from reasoning alone.\n"
            "Do not use a tool when you can answer directly and reliably "
            "from general knowledge (non-math facts). Never use web_search "
            "for arithmetic.\n\n"
            "For any question that required web_search (i.e. actual "
            "research, not a quick calculation or trivial fact), end your "
            "final response with exactly this structure:\n\n"
            "### Synthesis\n"
            "<your full reasoned answer, referencing what the sources say>\n\n"
            "### Conclusion\n"
            "<a one or two sentence bottom-line takeaway>\n\n"
            "For simple questions that did not need research (math, "
            "trivial facts), just answer directly without these markers."
        ),
        tools=[calculator, web_search],
    )


root_agent = build_agent()