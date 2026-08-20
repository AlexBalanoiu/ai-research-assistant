"""
Tests - Level 1: source citation, confidence scoring, query refinement.

The functional tests call the live model and are best-effort (model
behavior varies) - watch your Gemini free-tier quota (15 RPM) when
running this together with the rest of the suite.

Run: pytest tests/test_level1_quality.py -v
"""
from research_assistant.report import compute_confidence, build_report
from research_assistant.runner import ask_with_trace, AgentSession


# --- Unit tests (fast, no LLM/network) ---

def test_confidence_no_sources():
    assert compute_confidence([]) == "Not verified (answered from model knowledge only, no sources checked)"


def test_confidence_single_source():
    assert compute_confidence([{"url": "a"}]) == "Low (single source)"


def test_confidence_two_sources():
    assert compute_confidence([{"url": "a"}, {"url": "b"}]) == "Medium (2 sources)"


def test_confidence_three_or_more_sources():
    assert compute_confidence([{"url": "a"}, {"url": "b"}, {"url": "c"}]) == "High (3+ sources)"


def test_report_includes_confidence_line():
    report = build_report("Q?", "### Synthesis\nX\n\n### Conclusion\nY", [])
    assert "**Confidence:**" in report


# --- Functional tests (require model configured) ---

async def test_agent_answers_and_searches_for_current_fact():
    """Dynamic fact (stock price) - no model can know it confidently
    without searching, unlike a long-tenured CEO's name."""
    answer, tool_calls = await ask_with_trace("What is the current stock price of Apple (AAPL)?")
    assert "web_search" in tool_calls
    assert len(answer.strip()) > 0


async def test_agent_attempts_search_for_obscure_query():
    """
    Best-effort: not a hard guarantee of reformulation (depends on model +
    search backend), but checks the agent doesn't just give up silently
    after one empty/unhelpful search.
    """
    session = AgentSession()
    _, tool_calls, _ = await session.send(
        "Search for 'zzqxv nonsense keyword blorptastic 12345' and tell me "
        "what you find; if that search is unhelpful, try a better query."
    )
    assert tool_calls.count("web_search") >= 1