
from research_assistant.tools.budget import check_and_consume, MAX_TOOL_CALLS_PER_TURN
from research_assistant.tools.calculator import calculator
from research_assistant.fact_check import fact_check
from tests.fakes import FakeToolContext


# --- Unit tests (fast, no LLM/network) ---

def test_budget_allows_up_to_the_limit():
    ctx = FakeToolContext()
    for _ in range(MAX_TOOL_CALLS_PER_TURN):
        assert check_and_consume(ctx) is True


def test_budget_blocks_after_the_limit():
    ctx = FakeToolContext()
    for _ in range(MAX_TOOL_CALLS_PER_TURN):
        check_and_consume(ctx)
    assert check_and_consume(ctx) is False


def test_calculator_returns_error_once_budget_is_exhausted():
    ctx = FakeToolContext()
    for _ in range(MAX_TOOL_CALLS_PER_TURN):
        calculator("1+1", ctx)
    result = calculator("1+1", ctx)
    assert "error" in result
    assert "budget" in result["error"].lower()


async def test_fact_check_returns_not_applicable_without_sources():
    result = await fact_check("Some synthesis text.", sources=[])
    assert result.startswith("VERDICT: NOT APPLICABLE")


# --- Functional test (requires model configured, 1 live LLM call) ---

async def test_fact_check_flags_supported_synthesis():
    sources = [{"title": "Example", "url": "https://example.com", "snippet": "The sky is blue."}]
    result = await fact_check("The sky is blue.", sources)
    assert result.startswith("VERDICT:")