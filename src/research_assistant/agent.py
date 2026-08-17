"""
Step 3 - Agent + calculator + web search tools.
Verifies correct tool selection across math vs factual/current questions.
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
        description="Research agent - step 3: reasoning + calculator + web search",
        instruction=(
            "You are a research assistant with two tools:\n"
            "- calculator: use for any arithmetic or math expression.\n"
            "- web_search: use for current events, facts about specific "
            "entities, prices, or anything you cannot be certain about "
            "from reasoning alone.\n"
            "Do not use a tool when you can answer directly and reliably "
            "from general knowledge (e.g. well-known static facts). "
            "Never use calculator for non-math questions or web_search for "
            "simple arithmetic."
        ),
        tools=[calculator, web_search],
    )


root_agent = build_agent()