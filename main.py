"""
CLI pentru agentul de cercetare - o singura intrebare sau chat interactiv.

Utilizare:
    python main.py "Cine e actualul CEO al OpenAI?"   # o singura intrebare
    python main.py                                     # chat interactiv (follow-up-uri)
"""
import asyncio
import re
import sys
from datetime import datetime
from pathlib import Path

from research_assistant.runner import AgentSession, generate_report, report_from_session

REPORTS_DIR = Path(__file__).parent / "reports"


def _slugify(text: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug[:max_len].strip("-") or "report"


def save_report_file(question: str, report: str) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}_{_slugify(question)}.md"
    path = REPORTS_DIR / filename
    path.write_text(report, encoding="utf-8")
    return path


def _print_saved_location(path: Path) -> None:
    windows_path = str(path.resolve()).replace(
        "/home/", r"\\wsl$\Ubuntu\home\\"
    ).replace("/", "\\")
    print(f"\n---\nSalvat: {path.resolve()}")
    print(f"Din Windows Explorer: {windows_path}\n")


async def run_single(question: str) -> None:
    report = await generate_report(question)
    print(report)
    _print_saved_location(save_report_file(question, report))


async def run_chat() -> None:
    print("Chat de cercetare - poti pune intrebari de follow-up in aceeasi sesiune.")
    print("Scrie 'exit' sau 'quit' ca sa iesi.\n")
    session = AgentSession()
    while True:
        question = input("Intrebare: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        report = await report_from_session(session, question)
        print(report)
        _print_saved_location(save_report_file(question, report))


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if question:
        asyncio.run(run_single(question))
    else:
        asyncio.run(run_chat())


if __name__ == "__main__":
    main()