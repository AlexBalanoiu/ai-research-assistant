"""
Helpers for running the agent outside the ADK CLI/web UI (scripts, tests).
"""
import asyncio
import re
import uuid

from litellm.exceptions import RateLimitError
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

from research_assistant.agent import root_agent
from research_assistant.report import build_report

APP_NAME = "research_assistant"
_MAX_RETRIES = 2
_DEFAULT_RETRY_DELAY = 20.0


def _extract_retry_delay(error_message: str) -> float:
    """Parses "Please retry in 45.7s" out of the Gemini 429 error message."""
    match = re.search(r"retry in ([\d.]+)s", error_message)
    return float(match.group(1)) + 1.0 if match else _DEFAULT_RETRY_DELAY


class AgentSession:
    """Wraps a single ADK session so multiple turns can share state/memory."""

    def __init__(self):
        self.runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
        self.user_id = "local_user"
        self.session_id = str(uuid.uuid4())
        self._created = False

    async def _ensure_session(self):
        if not self._created:
            await self.runner.session_service.create_session(
                app_name=APP_NAME, user_id=self.user_id, session_id=self.session_id
            )
            self._created = True

    async def send(self, question: str) -> tuple[str, list[str], list[dict]]:
        """Sends one turn, retrying on rate limits. Returns (final_text, tool_names_called, tool_results)."""
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return await self._send_once(question)
            except RateLimitError as exc:
                if attempt == _MAX_RETRIES:
                    raise
                delay = _extract_retry_delay(str(exc))
                print(f"[rate limit] waiting {delay:.0f}s before retry ({attempt + 1}/{_MAX_RETRIES})...")
                await asyncio.sleep(delay)

    async def _send_once(self, question: str) -> tuple[str, list[str], list[dict]]:
        await self._ensure_session()
        content = Content(role="user", parts=[Part(text=question)])
        final_text = ""
        tool_calls: list[str] = []
        tool_results: list[dict] = []

        async for event in self.runner.run_async(
            user_id=self.user_id, session_id=self.session_id, new_message=content
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if getattr(part, "function_call", None):
                        tool_calls.append(part.function_call.name)
                    if getattr(part, "function_response", None):
                        tool_results.append(part.function_response.response)
            if event.is_final_response() and event.content and event.content.parts:
                final_text = event.content.parts[0].text

        return final_text, tool_calls, tool_results


async def ask(question: str) -> str:
    """One-off question, fresh session. Returns only the final text answer."""
    text, _, _ = await AgentSession().send(question)
    return text


async def ask_with_trace(question: str) -> tuple[str, list[str]]:
    """One-off question, fresh session. Returns (final_text, tool_names_called)."""
    text, tool_calls, _ = await AgentSession().send(question)
    return text, tool_calls


async def generate_report(question: str) -> str:
    """One-off question, fresh session. Returns a formatted markdown report."""
    answer, _, tool_results = await AgentSession().send(question)
    return build_report(question, answer, tool_results)