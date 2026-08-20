"""
Step 5 - Markdown report generator.
Structure: question -> sources -> synthesis -> conclusion.

Sources are extracted programmatically from actual web_search tool
results (not trusted from the LLM's prose, to avoid hallucinated URLs).
Synthesis/Conclusion are parsed from the agent's final answer, which is
instructed (see agent.py) to mark them with "### Synthesis" / "### Conclusion".
"""
import re


def build_report(question: str, answer: str, tool_results: list[dict]) -> str:
    sources = _extract_sources(tool_results)
    synthesis, conclusion = _split_answer(answer)

    lines = [
        "# Research Report",
        "",
        f"**Question:** {question}",
        "",
        "## Sources",
    ]
    if sources:
        lines += [f"- [{s.get('title') or s.get('url')}]({s.get('url')})" for s in sources]
    else:
        lines.append("_No external sources were used for this answer._")

    lines += [
        "",
        "## Synthesis",
        synthesis,
        "",
        "## Conclusion",
        conclusion,
    ]
    return "\n".join(lines)


def _extract_sources(tool_results: list[dict]) -> list[dict]:
    sources = []
    seen_urls = set()
    for result in tool_results:
        for item in result.get("results", []):
            url = item.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append(item)
    return sources


def _split_answer(answer: str) -> tuple[str, str]:
    synthesis_match = re.search(
        r"###\s*Synthesis\s*(.*?)(?=###\s*Conclusion|\Z)", answer, re.S | re.I
    )
    conclusion_match = re.search(r"###\s*Conclusion\s*(.*)", answer, re.S | re.I)

    if synthesis_match and conclusion_match:
        return synthesis_match.group(1).strip(), conclusion_match.group(1).strip()

    # Fallback: agent didn't use the markers (e.g. trivial question) -
    # treat the whole answer as the synthesis.
    return answer.strip(), "(no explicit conclusion provided)"