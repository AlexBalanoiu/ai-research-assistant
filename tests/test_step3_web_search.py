"""
Functional tests - Step 3: web search tool + tool selection across two tools.

Run: pytest tests/test_step3_web_search.py -v
"""
from research_assistant.tools.web_search import web_search
from research_assistant.runner import ask_with_trace
from tests.fakes import FakeToolContext


# --- Unit test (fast, no LLM, needs internet) ---

def test_web_search_pure_function_returns_results():
    result = web_search("Python programming language", FakeToolContext(), max_results=3)
    assert "results" in result
    assert len(result["results"]) > 0
    assert "url" in result["results"][0]


# --- Functional tests (require Ollama/Gemini configured) ---

async def test_agent_uses_web_search_for_current_events():
    """Correct tool selection: factual/current question -> web_search, not calculator."""
    _, tool_calls = await ask_with_trace(
        "Who is the current CEO of Microsoft?"
    )
    assert "web_search" in tool_calls
    assert "calculator" not in tool_calls


async def test_agent_still_uses_calculator_for_math():
    """
    Regression check: adding web_search must not break math handling.
    Primary check is correctness of the answer. Calling the calculator
    tool is instructed but not 100% guaranteed by any LLM (models can
    solve trivial arithmetic mentally despite instructions) - what must
    never happen is misusing web_search for math.
    """
    answer, tool_calls = await ask_with_trace("What is 88 * 7?")
    assert "616" in answer
    assert "web_search" not in tool_calls


async def test_agent_does_not_search_for_arithmetic_phrased_as_question():
    """Tool misuse check: math phrased as a question should not trigger search."""
    _, tool_calls = await ask_with_trace("What is 15% of 200?")
    assert "web_search" not in tool_calls