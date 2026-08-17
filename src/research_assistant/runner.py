"""
Helper for agent running  ADK CLI/web 
"""
import uuid
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

from research_assistant.agent import root_agent

APP_NAME = "research_assistant"


async def ask(question: str) -> str:
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    user_id = "local_user"
    session_id = str(uuid.uuid4())

    await runner.session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )

    content = Content(role="user", parts=[Part(text=question)])
    final_text = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text

    return final_text