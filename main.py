"""
simple CLI

use:
    python main.py "actual question"
    python main.py                      # interactive question
"""
import asyncio
import re
import sys
from datetime import datetime
from pathlib import Path

from research_assistant.runner import generate_report

REPORTS_DIR = Path(__file__).parent / "reports"


def _slugify(text: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug[:max_len].strip("-") or "report"


def save_report(question: str, report: str) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}_{_slugify(question)}.md"
    path = REPORTS_DIR / filename
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or input("Intrebare: ").strip()
    if not question:
        print("No question asked")
        return

    report = asyncio.run(generate_report(question))
    print(report)

    saved_path = save_report(question, report)
    windows_path = str(saved_path.resolve()).replace(
        "/home/", r"\\wsl$\Ubuntu\home\\"
    ).replace("/", "\\")

    print(f"\n---\nSaved at: {saved_path.resolve()}")
    print(f"From Explorer: {windows_path}")


if __name__ == "__main__":
    main()