"""Data shapes and deterministic grading for quiz mode.

The division of labour here is deliberate. Three of QuizResult's five fields
are facts the app already knows exactly - the score, the count, the clock - so
the app computes them. Only the two that need judgment are asked of the model.
Letting a model report a score it did not compute is how you get a confident,
wrong number, the same failure the calculator tool exists to prevent.

Grading follows the same principle. Numeric answers are graded here, in Python,
outside the model's reach. We already proved a model can ignore a tool result
it dislikes; a grader that can be argued with is worse than useless for exam
prep, because a student who talks their way to "correct" learns nothing.
"""

import math
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from study_helper.tools import CalculationError, evaluate_expression


class QuizQuestion(BaseModel):
    """One generated question, with enough metadata for the app to grade it."""

    question: str
    expected_answer: str
    topic: str = Field(description="Narrow sub-topic, for weak-area reporting.")
    numeric: bool = Field(
        description="True when the answer is a number or expression that can be "
        "checked exactly, false when it needs judgment."
    )


class GradeResult(BaseModel):
    """A model's judgment of an open-ended answer."""

    correct: bool
    feedback: str


class QuizAssessment(BaseModel):
    """The judgment-only slice the model is asked for at the end."""

    weak_topics: list[str] = Field(default_factory=list)
    verdict: str


class QuizResult(BaseModel):
    """The final summary of a quiz session."""

    score_percent: float = Field(ge=0, le=100)
    weak_topics: list[str] = Field(default_factory=list)
    questions_answered: int = Field(ge=0)
    time_used_seconds: int = Field(ge=0)
    verdict: str


@dataclass(slots=True)
class Answer:
    """One graded exchange, recorded by the app as the quiz runs."""

    question: str
    topic: str
    expected: str
    given: str
    correct: bool
    feedback: str


@dataclass(slots=True)
class NumericGrade:
    correct: bool
    reason: str


# Trailing units and words: "9.8 m/s^2" -> "9.8", "42 apples" -> "42".
_TRAILING_WORDS = re.compile(r"[a-zA-Z_][a-zA-Z0-9_^/*\s.-]*$")


def _to_number(text: str) -> float | None:
    """Parse a student's answer into a number, tolerating common noise.

    Reuses the calculator's safe evaluator so equivalent forms compare equal:
    "1/2", "0.5" and "2**-1" are all 0.5.
    """
    cleaned = text.strip().replace(",", "").replace("$", "")
    cleaned = cleaned.removeprefix("=").strip()
    if not cleaned:
        return None

    for candidate in (cleaned, _TRAILING_WORDS.sub("", cleaned).strip()):
        if not candidate:
            continue
        try:
            return evaluate_expression(candidate)
        except CalculationError:
            continue
    return None


def grade_numeric_answer(given: str, expected: str, tolerance: float = 1e-6) -> NumericGrade:
    """Grade a numeric answer deterministically. Never raises."""
    expected_value = _to_number(expected)
    if expected_value is None:
        return NumericGrade(False, f"could not read the expected answer {expected!r}")

    given_value = _to_number(given)
    if given_value is None:
        return NumericGrade(False, f"expected a number, got {given.strip()!r}")

    if math.isclose(given_value, expected_value, rel_tol=tolerance, abs_tol=tolerance):
        return NumericGrade(True, "matches the expected value")
    return NumericGrade(False, f"expected {expected_value:g}, got {given_value:g}")


def summarize(answers: list[Answer], elapsed_seconds: float, assessment: QuizAssessment) -> QuizResult:
    """Assemble the final result: app-computed numbers, model-supplied judgment."""
    answered = len(answers)
    correct = sum(1 for answer in answers if answer.correct)
    return QuizResult(
        score_percent=round(100 * correct / answered, 1) if answered else 0.0,
        weak_topics=assessment.weak_topics,
        questions_answered=answered,
        time_used_seconds=int(elapsed_seconds),
        verdict=assessment.verdict,
    )


def missed_topics(answers: list[Answer]) -> list[str]:
    """Topics of the questions actually answered wrong, in order, deduplicated."""
    seen: dict[str, None] = {}
    for answer in answers:
        if not answer.correct:
            seen.setdefault(answer.topic, None)
    return list(seen)
