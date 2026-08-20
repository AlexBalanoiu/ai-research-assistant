"""
Markdown report generator.
Structure: question -> confidence -> sources -> synthesis -> conclusion.

Sources are extracted programmatically from actual web_search tool
results (not trusted from the LLM's prose, to avoid hallucinated URLs).
Confidence is computed deterministically from the number of distinct
sources found (not self-reported by the LLM - more reliable/testable).
Synthesis/Conclusion are parsed from the agent's final answer, which is
instructed (see agent.py) to mark them with "### Synthesis" / "### Conclusion".
"""
import re

_CONFIDENCE_LEVELS = [
    (0, "Not verified (answered from model knowledge only, no sources checked)"),
    (1, "Low (single source)"),
    (2, "Medium (2 sources)"),
    (3, "High (3+ sources)"),
]


def build_report(question: str, answer: str, tool_results: list[dict]) -> str:
    sources = extract_sources(tool_results)
    synthesis, conclusion = split_answer(answer)
    confidence = compute_confidence(sources)

    lines = [
        "# Research Report",
        "",
        f"**Question:** {question}",
        f"**Confidence:** {confidence}",
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


def extract_sources(tool_results: list[dict]) -> list[dict]:
    sources = []
    seen_urls = set()
    for result in tool_results:
        for item in result.get("results", []):
            url = item.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append(item)
    return sources


def compute_confidence(sources: list[dict]) -> str:
    count = len(sources)
    for threshold, label in reversed(_CONFIDENCE_LEVELS):
        if count >= threshold:
            return label
    return _CONFIDENCE_LEVELS[0][1]


def split_answer(answer: str) -> tuple[str, str]:
    synthesis_match = re.search(
        r"###\s*Synthesis\s*(.*?)(?=###\s*Conclusion|\Z)", answer, re.S | re.I
    )
    conclusion_match = re.search(r"###\s*Conclusion\s*(.*)", answer, re.S | re.I)

    if synthesis_match and conclusion_match:
        return synthesis_match.group(1).strip(), conclusion_match.group(1).strip()

    return answer.strip(), "(no explicit conclusion provided)"