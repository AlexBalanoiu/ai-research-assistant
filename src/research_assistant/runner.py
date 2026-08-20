"""
Helpers for running the agent outside the ADK CLI/web UI (scripts, tests).

Includes automatic model fallback: if the primary model (Gemini) hits a
rate limit, this session switches to the local Ollama model
(FALLBACK_MODEL_ID) - the model-fallback decision from the project plan
(section 5.4). The fallback is "sticky": once triggered, the whole
AgentSession keeps using the fallback runner for the rest of the
conversation. A short retry is applied to fallback connection errors too
(Ollama can drop the connection on a cold start or under memory pressure).
"""
import asyncio
import os
import uuid

from litellm.exceptions import RateLimitError, APIConnectionError
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

from research_assistant.agent import build_agent, root_agent
from research_assistant.report import build_report, extract_sources, split_answer

APP_NAME = "research_assistant"
FALLBACK_MODEL_ID = os.environ.get("FALLBACK_MODEL_ID", "ollama_chat/llama3.1")
_FALLBACK_RETRIES = 2
_FALLBACK_RETRY_DELAY = 3.0


async def _run_turn(
    runner: InMemoryRunner, user_id: str, session_id: str, question: str
) -> tuple[str, list[str], list[dict]]:
    content = Content(role="user", parts=[Part(text=question)])
    final_text = ""
    tool_calls: list[str] = []
    tool_results: list[dict] = []

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
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


class AgentSession:
    """Wraps a single ADK session so multiple turns can share state/memory.

    If the primary model gets rate-limited, this session switches
    ("sticky") to a local fallback runner for the rest of its lifetime,
    reusing the SAME fallback session across turns.
    """

    def __init__(self):
        self.runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
        self.user_id = "local_user"
        self.session_id = str(uuid.uuid4())
        self._created = False

        self._fallback_runner: InMemoryRunner | None = None
        self._fallback_session_id: str | None = None
        self._using_fallback = False

    async def _ensure_session(self):
        if not self._created:
            await self.runner.session_service.create_session(
                app_name=APP_NAME, user_id=self.user_id, session_id=self.session_id
            )
            self._created = True

    async def _ensure_fallback_session(self):
        if self._fallback_runner is None:
            fallback_agent = build_agent(model_id=FALLBACK_MODEL_ID)
            self._fallback_runner = InMemoryRunner(agent=fallback_agent, app_name=APP_NAME)
            self._fallback_session_id = f"{self.session_id}-fallback"
            await self._fallback_runner.session_service.create_session(
                app_name=APP_NAME, user_id=self.user_id, session_id=self._fallback_session_id
            )

    async def send(self, question: str) -> tuple[str, list[str], list[dict]]:
        """Sends one turn. Falls back to a local model if the primary
        model is rate-limited, and stays on the fallback for the rest of
        this session once that happens."""
        if self._using_fallback:
            return await self._send_fallback(question)

        await self._ensure_session()
        try:
            return await _run_turn(self.runner, self.user_id, self.session_id, question)
        except RateLimitError as exc:
            print(f"[fallback] primary model rate-limited ({exc}); switching this session to {FALLBACK_MODEL_ID}...")
            self._using_fallback = True
            return await self._send_fallback(question)

    async def _send_fallback(self, question: str) -> tuple[str, list[str], list[dict]]:
        await self._ensure_fallback_session()
        last_error = None
        for attempt in range(_FALLBACK_RETRIES + 1):
            try:
                return await _run_turn(
                    self._fallback_runner, self.user_id, self._fallback_session_id, question
                )
            except APIConnectionError as exc:
                last_error = exc
                if attempt < _FALLBACK_RETRIES:
                    print(f"[fallback] {FALLBACK_MODEL_ID} connection dropped, retrying "
                          f"({attempt + 1}/{_FALLBACK_RETRIES}) - is 'ollama serve' running?")
                    await asyncio.sleep(_FALLBACK_RETRY_DELAY)
        raise RuntimeError(
            f"Both the primary model and the fallback ({FALLBACK_MODEL_ID}) failed. "
            f"Check that 'ollama serve' is running and the model is pulled. Last error: {last_error}"
        ) from last_error


async def ask(question: str) -> str:
    """One-off question, fresh session. Returns only the final text answer."""
    text, _, _ = await AgentSession().send(question)
    return text


async def ask_with_trace(question: str) -> tuple[str, list[str]]:
    """One-off question, fresh session. Returns (final_text, tool_names_called)."""
    text, tool_calls, _ = await AgentSession().send(question)
    return text, tool_calls


async def report_from_session(
    session: AgentSession, question: str, run_fact_check: bool = False
) -> str:
    """Builds a report using an EXISTING session, and saves it to persistent history."""
    answer, _, tool_results = await session.send(question)
    report = build_report(question, answer, tool_results)

    sources = extract_sources(tool_results)
    synthesis, _ = split_answer(answer)

    from research_assistant.memory.report_history import save_report as save_history
    save_history(question, synthesis, sources)

    if run_fact_check:
        from research_assistant.fact_check import fact_check
        critique = await fact_check(synthesis, sources)
        report += f"\n\n## Fact-Check\n{critique}"

    return report


async def generate_report(question: str, run_fact_check: bool = False) -> str:
    """One-off question, fresh session. Returns a formatted markdown report."""
    return await report_from_session(AgentSession(), question, run_fact_check)