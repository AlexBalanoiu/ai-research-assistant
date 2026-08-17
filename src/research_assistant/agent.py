"""
Etapa 1 — minimal agent no tools dirrect answer
only llm based reasoning 
"""
import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm


def build_agent() -> Agent:
    model_id = os.environ.get("MODEL_ID", "ollama_chat/llama3.1")

    return Agent(
        model=LiteLlm(model=model_id),
        name="research_assistant_v1",
        description="Research agent — step 1",
        instruction=(
            "You are a research agent. You answer questions dirrectly, "
            "concisely and on point. If you are not sure about a fact you say that "
            "you cand verify the information, you have no tools yet"
        ),
    )


root_agent = build_agent()