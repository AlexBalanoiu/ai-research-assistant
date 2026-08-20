"""
Level 2 - Fact-checking pass. A second, independent LLM call reviews the
Synthesis against the extracted Sources and flags unsupported claims.

Kept separate/opt-in (see runner.generate_report(run_fact_check=True))
since it doubles the LLM calls for a single question - relevant given
free-tier rate limits.
"""
import os

from dotenv import load_dotenv
from litellm import acompletion

load_dotenv()

_FACT_CHECK_PROMPT = """You are a fact-checking reviewer. Below are a set of \
sources and a synthesis written from them. Check whether the synthesis is \
actually supported by the sources.

Sources:
{sources}

Synthesis:
{synthesis}

Respond in this exact format:
VERDICT: SUPPORTED or VERDICT: ISSUES FOUND
Then, on the next line, a one or two sentence explanation. If ISSUES FOUND, \
name the specific unsupported claim(s)."""


async def fact_check(synthesis: str, sources: list[dict]) -> str:
    """Returns the raw critique text, starting with a VERDICT line."""
    if not sources:
        return (
            "VERDICT: NOT APPLICABLE\n"
            "No external sources were used, so there is nothing to fact-check against."
        )

    sources_text = "\n".join(
        f"- {s.get('title') or s.get('url')}: {s.get('snippet', '')}" for s in sources
    )
    model_id = os.environ.get("MODEL_ID", "ollama_chat/llama3.1")

    response = await acompletion(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": _FACT_CHECK_PROMPT.format(
                    sources=sources_text, synthesis=synthesis
                ),
            }
        ],
    )
    return response.choices[0].message.content.strip()