"""Output guardrail: catch a tutor that solved its own practice question.

The tutors are told to leave the "Check yourself" question open, but that is
an instruction, not a guarantee - models leak the answer anyway, sometimes a
wrong one. This module detects that and produces a repaired answer.

A guardrail cannot rewrite the agent's output; it can only observe and trip a
tripwire. So the repaired text rides along in `GuardrailFunctionOutput.output_info`
and the caller (see `study_helper.cli`) prints that instead.
"""

import re
from dataclasses import dataclass

from agents import Agent, GuardrailFunctionOutput, RunContextWrapper, output_guardrail

PLACEHOLDER = "(Solve this yourself using the steps above.)"

# The "Check yourself" heading, whatever markdown the model wrapped it in.
# It must stop at the end of the heading itself: models sometimes put the
# practice question on the SAME line, and swallowing the rest of the line
# would leave nothing to inspect.
_HEADING = re.compile(
    r"^[ \t>*#-]*\**\s*check\s*yourself\b\s*\**\s*:?",
    re.IGNORECASE | re.MULTILINE,
)

# Phrases a tutor uses immediately before leaking the answer.
_LEAK_PHRASE = re.compile(
    r"\b(?:see if you get|check if you get|you should get|you'll get|"
    r"the answer is|answer\s*[:=]|it should be|which equals|equals)\b",
    re.IGNORECASE,
)

# A dangling conjunction left behind after cutting at a leak phrase.
_DANGLING = re.compile(r"[\s,;:]*\b(?:and|then|to see if)\s*$", re.IGNORECASE)

_HAS_DIGIT = re.compile(r"\d")


@dataclass(slots=True)
class CheckYourselfReport:
    """What the guardrail found, plus the repaired answer if it found a leak."""

    leaked: bool
    evidence: str = ""
    cleaned: str = ""


def _drop_unclosed_bracket(text: str) -> str:
    """Cut a trailing "(Try it and..." left dangling when we truncate mid-bracket."""
    open_positions: list[int] = []
    for index, char in enumerate(text):
        if char == "(":
            open_positions.append(index)
        elif char == ")" and open_positions:
            open_positions.pop()
    return text[: open_positions[0]].rstrip() if open_positions else text


def repair_check_yourself(text: str) -> CheckYourselfReport:
    """Find a leaked answer in the Check Yourself section and cut it out.

    Two signals, because one is not enough:

    1. A leak phrase ("see if you get 37864"). The digits in a practice
       question are legitimate, so position alone cannot distinguish the
       problem from its answer - the giveaway is the phrasing.
    2. Digits after the final question mark ("...? Answer: 37873565"),
       which catches the trailing-answer shape.

    Pure function, no model calls, so it is cheap and directly testable.
    """
    heading = _HEADING.search(text)
    if not heading:
        return CheckYourselfReport(leaked=False)

    body = text[heading.end() :]

    # A leak phrase only counts when an actual number follows it. "see if you
    # get the same result as a calculator" gives nothing away; "see if you get
    # 37864" does.
    cut = next(
        (m.start() for m in _LEAK_PHRASE.finditer(body) if _HAS_DIGIT.search(body[m.end() :])),
        None,
    )
    if cut is None:
        mark = body.rfind("?")
        if mark == -1 or not _HAS_DIGIT.search(body[mark + 1 :]):
            return CheckYourselfReport(leaked=False)
        cut = mark + 1

    kept = _DANGLING.sub("", body[:cut].rstrip().rstrip(",;:-"))
    kept = _drop_unclosed_bracket(kept)
    if not kept.endswith(("?", ".", "!")):
        kept += "."

    return CheckYourselfReport(
        leaked=True,
        evidence=body[cut:].strip()[:120],
        cleaned=f"{text[: heading.end()]}{kept} {PLACEHOLDER}\n",
    )


@output_guardrail
async def no_leaked_answer(
    ctx: RunContextWrapper, agent: Agent, output: str
) -> GuardrailFunctionOutput:
    """Trip when a tutor solved the practice question it was told to leave open."""
    report = repair_check_yourself(output)
    return GuardrailFunctionOutput(
        output_info=report,
        tripwire_triggered=report.leaked,
    )
