"""
Functional tests - Step 2: calculator tool.

Run: pytest tests/test_step2_calculator.py -v
"""
from research_assistant.tools.calculator import calculator
from research_assistant.runner import ask_with_trace


# --- Unit test (fast, no LLM needed) ---

def test_calculator_pure_function():
    assert calculator("2 + 2") == {"result": 4}
    assert calculator("10 / 4") == {"result": 2.5}
    assert "error" in calculator("import os")  # not a valid arithmetic expr


# --- Functional tests (require Ollama running) ---

async def test_agent_uses_calculator_for_math():
    """Verifies the agent actually calls the tool, not just guesses the answer."""
    answer, tool_calls = await ask_with_trace(
        "What is 4837 * 291? Reply with just the number."
    )
    assert "calculator" in tool_calls
    assert "1407567" in answer.replace(",", "").replace(" ", "")


async def test_agent_skips_calculator_for_non_math():
    """Correct tool selection: no calculator call for a non-math question."""
    _, tool_calls = await ask_with_trace("What is the capital of France?")
    assert "calculator" not in tool_calls


async def test_agent_uses_calculator_for_division():
    answer, tool_calls = await ask_with_trace("What is 144 divided by 12?")
    assert "calculator" in tool_calls
    assert "12" in answer