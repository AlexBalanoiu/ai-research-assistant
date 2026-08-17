"""
Helpers for running the agent outside the ADK CLI/web UI (scripts, tests).
"""
import uuid
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

from research_assistant.agent import root_agent

APP_NAME = "research_assistant"


async def _new_session(runner: InMemoryRunner) -> tuple[str, str]:
    user_id = "local_user"
    session_id = str(uuid.uuid4())
    await runner.session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    return user_id, session_id


async def ask(question: str) -> str:
    """Returns only the agent's final text answer."""
    text, _ = await ask_with_trace(question)
    return text


async def ask_with_trace(question: str) -> tuple[str, list[str]]:
    """Returns (final_text, list_of_tool_names_called) - for asserting tool use in tests."""
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    user_id, session_id = await _new_session(runner)

    content = Content(role="user", parts=[Part(text=question)])
    final_text = ""
    tool_calls: list[str] = []

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "function_call", None):
                    tool_calls.append(part.function_call.name)
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text

    return final_text, tool_calls