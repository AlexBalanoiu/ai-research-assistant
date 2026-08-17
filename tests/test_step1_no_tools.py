"""


Requires ollama turned on localy :
    ollama serve
    ollama pull llama3.1

Run : pytest tests/test_step1_no_tools.py -v
"""
from research_assistant.runner import ask


async def test_agent_returns_nonempty_answer():
    """The agent has to give a text answer, not emplty"""
    answer = await ask("What is phyotosynthesys? summarised")
    assert answer and len(answer.strip()) > 0


async def test_agent_handles_simple_reasoning():
    """Simple reasoing without tool"""
    answer = await ask("What is 30 + 12? Answer only with the number ")
    assert "42" in answer


async def test_agent_does_not_crash_on_live_data_question():
    """
    Without tools, the agent can't check live data,
    We don't validate the answer( there is no way it's right)
    """
    answer = await ask("What is the USDT/BTC exchange rate right now")
    assert len(answer.strip()) > 0