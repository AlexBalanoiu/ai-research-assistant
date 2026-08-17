"""
Step 2 - Agent + calculator tool.
Verifies multi-step reasoning and correct tool selection.
"""
import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from research_assistant.tools.calculator import calculator


def build_agent() -> Agent:
    model_id = os.environ.get("MODEL_ID", "ollama_chat/llama3.1")

    return Agent(
        model=LiteLlm(model=model_id),
        name="research_assistant_v1",
        description="Research agent - step 2: reasoning + calculator tool",
        instruction=(
            "You are a research assistant. Answer questions directly and "
            "concisely. Use the calculator tool for any arithmetic or math "
            "expression instead of computing it yourself - it is more "
            "reliable. Do not use the calculator for non-math questions."
        ),
        tools=[calculator],
    )


root_agent = build_agent()