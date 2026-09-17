"""Interactive terminal chat with the study helper."""

import asyncio
import re
import sys

from agents import Runner, SQLiteSession
from agents.exceptions import OutputGuardrailTripwireTriggered

from study_helper.context import StudyContext
from study_helper.hooks import EventEmittingHooks, RunEvent, format_event
from study_helper.quiz_session import format_result, run_quiz
from study_helper.roles.orchestrator import build_orchestrator
from study_helper.turns import atomic_turn

BANNER = """
Study Helper  -  ask a math or science question.
Type 'quiz' to start a timed quiz, or 'quit' to leave.
"""

# The tutors are told to emit plain ASCII, but instructions are suggestions,
# not guarantees - models still leak typography. Normalizing here makes the
# rule actually hold. Escapes are spelled out because several of these
# characters are invisible or indistinguishable in an editor.
_ASCII_MAP = {
    "‘": "'", "’": "'",           # curly single quotes
    "“": '"', "”": '"',           # curly double quotes
    "‐": "-", "‑": "-",           # hyphen, non-breaking hyphen
    "–": "-", "−": "-",           # en dash, minus sign
    "×": "*", "·": "*",           # multiplication sign, middle dot
    "•": "-", "…": "...",         # bullet, ellipsis
    "≈": "~", "≠": "!=",          # almost equal, not equal
    "≤": "<=", "≥": ">=",         # less/greater or equal
    "→": "->", "←": "<-",         # arrows
    "⇒": "=>",
    "√": "sqrt", "°": "deg",      # radical, degree sign
    "¹": "^1", "²": "^2", "³": "^3",
    "⁰": "^0", "⁴": "^4", "⁵": "^5",
    "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3",
    "₄": "4", "₅": "5", "₆": "6", "₇": "7",
    "₈": "8", "₉": "9",
}

# Exotic spaces: narrow no-break, no-break, thin, en, em.
_ODD_SPACES = "     "
# Digit grouping, thin-space or comma style: "2 278" / "7,006,652" -> "2278".
# The comma needs exactly three following digits so a list like "1,2,3"
# is not silently glued into "123".
_DIGIT_SEPARATOR = re.compile(rf"(?<=\d)(?:[{_ODD_SPACES}]|,(?=\d{{3}}(?!\d)))(?=\d)")
# An em dash joins clauses, so as a hyphen it needs spaces either side,
# otherwise "the answer—just try it" collapses to "the answer-just try it".
_EM_DASH = re.compile(r"\s*—\s*")


def to_ascii(text: str) -> str:
    """Replace the unicode typography models emit despite being told not to."""
    text = _DIGIT_SEPARATOR.sub("", text)
    text = _EM_DASH.sub(" - ", text)
    for fancy, plain in _ASCII_MAP.items():
        text = text.replace(fancy, plain)
    return re.sub(f"[{_ODD_SPACES}]", " ", text)


def print_event(event: RunEvent) -> None:
    """The terminal's consumer of run events. A UI supplies its own instead."""
    line = format_event(event)
    if line:
        print(to_ascii(line))


async def _ask(prompt: str, seconds_left: float) -> str:
    """Terminal answer collection.

    `seconds_left` is ignored on purpose. A blocking input() cannot be
    cancelled cleanly - asyncio.wait_for would return while the thread still
    waits for Enter, and that stray keystroke would land in the next prompt.
    So the terminal keeps its existing behaviour: the budget is checked
    between questions, not during one. The web UI enforces it properly.
    """
    return (await asyncio.to_thread(input, prompt)).strip()


async def _say(message: str) -> None:
    print(to_ascii(message))


async def quiz_mode() -> None:
    """Collect a topic and a time budget, then run the quiz."""
    topic = await _ask("\nquiz topic > ", 0)
    if not topic:
        print("no topic, no quiz.")
        return

    raw = await _ask("time budget in minutes > ", 0)
    try:
        minutes = float(raw)
        if minutes <= 0:
            raise ValueError
    except ValueError:
        print(f"{raw!r} is not a number of minutes, using 5.")
        minutes = 5.0

    print(f"\nRight - {minutes:g} minutes on {topic}. Type 'done' any time to stop early.")
    try:
        result = await run_quiz(topic, minutes * 60, ask=_ask, say=_say)
    except (EOFError, KeyboardInterrupt):
        print("\nquiz abandoned.")
        return
    print(to_ascii(format_result(result)))


async def chat() -> None:
    # Every turn starts here, so the student can switch subjects freely.
    front_desk = build_orchestrator()
    # No db_path => history lives in memory and vanishes when you quit.
    session = SQLiteSession("study-helper")

    print(BANNER)
    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            return

        if not question:
            continue
        if question.lower() in {"quit", "exit", ":q"}:
            print("bye!")
            return
        if question.lower() == "quiz":
            await quiz_mode()
            continue

        try:
            # A failed turn must leave nothing behind, or its question gets
            # re-sent next time and answered instead of the new one.
            async with atomic_turn(session):
                try:
                    result = await Runner.run(
                        front_desk,
                        question,
                        session=session,
                        hooks=EventEmittingHooks(print_event),
                        # Fresh per turn: the routing reason is this run's only.
                        context=StudyContext(),
                    )
                except OutputGuardrailTripwireTriggered as tripped:
                    # A guardrail cannot rewrite the output, so it hands us a
                    # repaired version in output_info. Handled here rather than
                    # outside, because this turn succeeded - the reply just
                    # needed cleaning - so it must not be rolled back.
                    report = tripped.guardrail_result.output.output_info
                    answer = to_ascii(report.cleaned)
                    author = tripped.guardrail_result.agent.name
                    note = f"   [guardrail removed a leaked answer: {report.evidence!r}]"
                    # The run aborted before the reply was stored; the question
                    # is already there, so add only the repaired reply.
                    await session.add_items([{"role": "assistant", "content": answer}])
                else:
                    answer = to_ascii(result.final_output)
                    author = result.last_agent.name
                    note = ""
        except Exception as err:  # bad key, rate limit, network, bad model name
            print(f"\n[error] {type(err).__name__}: {err}\n")
            continue

        # last_agent is whoever actually produced the answer, so a handoff
        # is visible in the prompt label.
        print(f"\n{author} > {answer}\n")
        if note:
            print(f"{note}\n")


def main() -> None:
    # Windows consoles default to cp1252, which crashes on the unicode
    # (curly quotes, narrow spaces) models routinely emit.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(chat())
