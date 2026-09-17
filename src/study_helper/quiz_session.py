"""The quiz loop: ask, grade, track the clock, summarize.

Kept out of cli.py so it can be tested without a terminal. Input and output
are injected, and the clock is a callable, so tests can drive a whole quiz
with scripted answers and a fake clock.
"""

import time
from collections.abc import Awaitable, Callable, Iterable

from agents import Runner, SQLiteSession

from study_helper.quiz import (
    Answer,
    QuizAssessment,
    QuizQuestion,
    QuizResult,
    grade_numeric_answer,
    missed_topics,
    summarize,
)
from study_helper.roles.quiz_agent import (
    build_assessment_agent,
    build_grader_agent,
    build_question_agent,
)

DONE_WORDS = {"done", "stop", "quit", "exit", "finish"}

# Both interfaces are async, so the loop can serve a blocking terminal and an
# event-driven web UI without either one duplicating it.
# ask(prompt, seconds_left) -> the answer, or None if the time ran out first.
Ask = Callable[[str, float], Awaitable[str | None]]
Say = Callable[[str], Awaitable[None]]


async def _next_question(session: SQLiteSession, topic: str, asked: list[str]) -> QuizQuestion:
    avoid = ""
    if asked:
        avoid = "\nDo not repeat any of these already-asked questions:\n- " + "\n- ".join(asked)
    result = await Runner.run(
        build_question_agent(),
        f"Ask the next question on: {topic}.{avoid}",
        session=session,
    )
    return result.final_output


async def _grade(question: QuizQuestion, given: str) -> tuple[bool, str]:
    """Deterministic where possible, model judgment only where necessary."""
    if question.numeric:
        grade = grade_numeric_answer(given, question.expected_answer)
        return grade.correct, grade.reason

    result = await Runner.run(
        build_grader_agent(),
        f"Question: {question.question}\n"
        f"Expected answer: {question.expected_answer}\n"
        f"Student wrote: {given}",
    )
    judged = result.final_output
    return judged.correct, judged.feedback


async def _assess(answers: Iterable[Answer]) -> QuizAssessment:
    """Ask the model to judge only what was missed. It never sees the score."""
    missed = [a for a in answers if not a.correct]
    if not missed:
        report = "The student answered every question correctly."
    else:
        report = "Questions the student got wrong:\n" + "\n".join(
            f"- [{a.topic}] {a.question} (expected {a.expected}, wrote {a.given})" for a in missed
        )
    result = await Runner.run(build_assessment_agent(), report)
    return result.final_output


async def run_quiz(
    topic: str,
    budget_seconds: float,
    ask: Ask,
    say: Say,
    clock: Callable[[], float] = time.monotonic,
) -> QuizResult:
    """Run a quiz until the time budget runs out or the student says done.

    `ask` is given the seconds remaining and may return None to mean "nobody
    answered in time". A terminal cannot interrupt a blocking input(), so it
    ignores the budget and never returns None; an event-driven UI honours it
    and the quiz ends even with the student idle.
    """
    started = clock()
    session = SQLiteSession("quiz")
    answers: list[Answer] = []
    asked: list[str] = []

    while True:
        remaining = budget_seconds - (clock() - started)
        if remaining <= 0:
            await say("\nTime's up.")
            break

        await say(f"\n({int(remaining)}s left)")
        try:
            question = await _next_question(session, topic, asked)
        except Exception as err:
            await say(f"[error] could not get a question: {type(err).__name__}: {err}")
            break

        asked.append(question.question)
        given = await ask(
            f"\nQ{len(answers) + 1}. {question.question}\nyour answer > ",
            budget_seconds - (clock() - started),
        )

        if given is None:
            await say("\nTime's up - no answer in time.")
            break
        if given.strip().lower() in DONE_WORDS:
            break
        # Check the clock again: the student may have spent the budget thinking.
        if clock() - started >= budget_seconds:
            await say("\nTime's up - that one doesn't count.")
            break

        correct, feedback = await _grade(question, given)
        answers.append(
            Answer(
                question=question.question,
                topic=question.topic,
                expected=question.expected_answer,
                given=given,
                correct=correct,
                feedback=feedback,
            )
        )
        await say(f"  {'correct' if correct else 'not quite'} - {feedback}")

    try:
        assessment = await _assess(answers)
    except Exception as err:
        # The score, count and clock are ours, so a model outage costs us only
        # the prose. Fall back to a computed summary rather than losing the quiz.
        await say(f"\n[note] no summary from the model ({type(err).__name__}); computing one.")
        assessment = fallback_assessment(answers)

    return summarize(answers, clock() - started, assessment)


def fallback_assessment(answers: list[Answer]) -> QuizAssessment:
    """A summary derived from the answers alone, for when the model is unavailable."""
    answered = len(answers)
    if not answered:
        return QuizAssessment(weak_topics=[], verdict="No questions answered.")

    correct = sum(1 for answer in answers if answer.correct)
    weak = missed_topics(answers)
    if not weak:
        verdict = f"All {answered} correct. Nothing to revise."
    else:
        verdict = f"{correct} of {answered} correct. Revise: {', '.join(weak)}."
    return QuizAssessment(weak_topics=weak, verdict=verdict)


def format_result(result: QuizResult) -> str:
    """Render the result for a terminal, rather than dumping JSON at the student."""
    minutes, seconds = divmod(result.time_used_seconds, 60)
    weak = ", ".join(result.weak_topics) if result.weak_topics else "none - nothing missed"
    return (
        "\n"
        "  QUIZ RESULT\n"
        "  -----------\n"
        f"  Score      : {result.score_percent:g}%\n"
        f"  Answered   : {result.questions_answered}\n"
        f"  Time used  : {minutes}m {seconds:02d}s\n"
        f"  Weak areas : {weak}\n"
        f"  Verdict    : {result.verdict}\n"
    )
