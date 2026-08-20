"""
Functional tests - Step 5: markdown report generator.

Run: pytest tests/test_step5_report.py -v
"""
from research_assistant.report import build_report
from research_assistant.runner import generate_report


# --- Unit tests (fast, no LLM/network) ---

def test_report_includes_question_and_sections():
    report = build_report(
        "What is Python?",
        "### Synthesis\nPython is a language.\n\n### Conclusion\nIt's popular.",
        tool_results=[],
    )
    assert "**Question:** What is Python?" in report
    assert "## Sources" in report
    assert "## Synthesis" in report
    assert "## Conclusion" in report


def test_report_extracts_and_dedupes_sources():
    tool_results = [
        {"results": [{"title": "Python.org", "url": "https://python.org", "snippet": "..."}]},
        {"results": [{"title": "Python.org", "url": "https://python.org", "snippet": "..."}]},
        {"results": [{"title": "Wikipedia", "url": "https://en.wikipedia.org/wiki/Python", "snippet": "..."}]},
    ]
    report = build_report("What is Python?", "### Synthesis\nX\n\n### Conclusion\nY", tool_results)
    assert report.count("https://python.org") == 1
    assert "https://en.wikipedia.org/wiki/Python" in report


def test_report_splits_synthesis_and_conclusion_when_markers_present():
    answer = "### Synthesis\nDetailed reasoning here.\n\n### Conclusion\nShort takeaway."
    report = build_report("Q?", answer, [])
    synthesis_idx = report.index("## Synthesis")
    conclusion_idx = report.index("## Conclusion")
    assert "Detailed reasoning here." in report[synthesis_idx:conclusion_idx]
    assert "Short takeaway." in report[conclusion_idx:]


def test_report_falls_back_gracefully_without_markers():
    """Trivial answers (e.g. from calculator) won't have the markers."""
    report = build_report("What is 2+2?", "The answer is 4.", [])
    assert "The answer is 4." in report
    assert "no explicit conclusion provided" in report


def test_report_notes_when_no_sources_used():
    report = build_report("What is 2+2?", "The answer is 4.", [])
    assert "No external sources were used" in report


# --- Functional test (requires model configured) ---

async def test_generate_report_end_to_end_for_research_question():
    """Uses a genuinely dynamic fact (stock price) that no model can know
    confidently without searching - avoids the flakiness of 'well-known
    but technically current' facts like long-tenured CEOs."""
    report = await generate_report("What is the current stock price of Apple (AAPL)?")
    assert report.startswith("# Research Report")
    assert "## Sources" in report
    assert "## Synthesis" in report
    assert "## Conclusion" in report
    assert "https://" in report