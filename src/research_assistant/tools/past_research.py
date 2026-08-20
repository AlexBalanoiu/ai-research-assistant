"""
 lets the agent check whether a similar question was
already researched previously (any past session), to reuse that work
instead of repeating web searches.
"""
from google.adk.tools import ToolContext

from research_assistant.memory.report_history import search_past_reports
from research_assistant.tools.budget import check_and_consume, MAX_TOOL_CALLS_PER_TURN


def check_past_research(topic: str, tool_context: ToolContext) -> dict:
    """Checks past research reports for a topic before searching the web.

    Call this FIRST for any research question, before web_search. If it
    returns relevant prior findings, reuse them directly (and mention
    they come from earlier research) instead of searching again. If
    nothing relevant comes back, proceed with web_search as normal.

    Args:
        topic: A short keyword or phrase describing the topic.
        tool_context: Injected automatically by ADK - session state access.

    Returns:
        A dict: {"matches": [{"question", "synthesis", "created_at"}, ...]}
        (empty list if nothing relevant), or {"error": <message>}.
    """
    if not check_and_consume(tool_context):
        return {
            "error": f"Tool call budget exceeded (max {MAX_TOOL_CALLS_PER_TURN} "
            "tool calls per question)."
        }
    try:
        return {"matches": search_past_reports(topic)}
    except Exception as exc:
        return {"error": f"Could not check past research: {exc}"}