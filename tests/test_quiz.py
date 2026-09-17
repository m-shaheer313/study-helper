"""Tests for quiz mode.

The loop is driven offline with scripted answers and a fake clock, so timer
and scoring behaviour is deterministic. Live tests cover the things only a
real model can show: that structured output parses, and that the handoff works.
"""

import pytest

from study_helper import quiz_session
from study_helper.quiz import (
    Answer,
    GradeResult,
    QuizAssessment,
    QuizQuestion,
    QuizResult,
    grade_numeric_answer,
    missed_topics,
    summarize,
)
from study_helper.quiz_session import format_result, run_quiz


class TestNumericGrading:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("4", "4"),
            ("0.5", "1/2"),          # equivalent forms
            ("1/2", "0.5"),
            ("2**3", "8"),
            (" 7006652 ", "7006652"),
            ("7,006,652", "7006652"),  # thousands separators
            ("9.8 m/s^2", "9.8"),      # trailing units
            ("= 12", "12"),
        ],
    )
    def test_accepts_equivalent_answers(self, given, expected):
        assert grade_numeric_answer(given, expected).correct

    @pytest.mark.parametrize(("given", "expected"), [("5", "4"), ("0.6", "1/2"), ("-3", "3")])
    def test_rejects_wrong_answers(self, given, expected):
        assert not grade_numeric_answer(given, expected).correct

    def test_non_numeric_answer_is_marked_wrong_not_crashed(self):
        grade = grade_numeric_answer("no idea", "42")
        assert not grade.correct
        assert "expected a number" in grade.reason

    def test_empty_answer_is_wrong(self):
        assert not grade_numeric_answer("", "42").correct

    def test_reason_explains_the_mismatch(self):
        assert "expected 4, got 5" in grade_numeric_answer("5", "4").reason


def _answer(topic: str, correct: bool) -> Answer:
    return Answer(
        question="q", topic=topic, expected="1", given="1", correct=correct, feedback=""
    )


class TestScoring:
    def test_score_is_computed_from_the_answers(self):
        answers = [_answer("a", True), _answer("b", True), _answer("c", False), _answer("d", False)]
        result = summarize(answers, 90.0, QuizAssessment(weak_topics=["c"], verdict="ok"))
        assert result.score_percent == 50.0
        assert result.questions_answered == 4
        assert result.time_used_seconds == 90

    def test_empty_quiz_scores_zero_without_dividing_by_zero(self):
        result = summarize([], 10.0, QuizAssessment(weak_topics=[], verdict="nothing attempted"))
        assert result.score_percent == 0.0
        assert result.questions_answered == 0

    def test_missed_topics_are_only_the_wrong_ones(self):
        answers = [_answer("algebra", True), _answer("calculus", False), _answer("algebra", False)]
        assert missed_topics(answers) == ["calculus", "algebra"]

    def test_missed_topics_deduplicates(self):
        assert missed_topics([_answer("x", False), _answer("x", False)]) == ["x"]

    def test_perfect_score_has_no_missed_topics(self):
        assert missed_topics([_answer("x", True)]) == []


async def _silent(_message: str) -> None:
    """An async `say` that discards output."""


def _from(answers):
    """An async `ask` that replays scripted answers, ignoring the clock."""

    async def ask(_prompt: str, _seconds_left: float) -> str:
        return next(answers)

    return ask


class FakeClock:
    """A clock the test advances by hand, so timing never depends on wall time."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def scripted(monkeypatch):
    """Replace the three model calls with deterministic stubs."""
    questions = iter(
        QuizQuestion(
            question=f"Question {n}?", expected_answer=str(n), topic=f"topic{n}", numeric=True
        )
        for n in range(1, 99)
    )

    async def fake_question(session, topic, asked):
        return next(questions)

    async def fake_assess(answers):
        return QuizAssessment(weak_topics=missed_topics(list(answers)), verdict="done")

    monkeypatch.setattr(quiz_session, "_next_question", fake_question)
    monkeypatch.setattr(quiz_session, "_assess", fake_assess)


class TestQuizLoop:
    async def test_produces_a_valid_quiz_result(self, scripted):
        clock = FakeClock()
        answers = iter(["1", "2", "done"])
        result = await run_quiz(
            "algebra", 600, ask=_from(answers), say=_silent, clock=clock
        )
        assert isinstance(result, QuizResult)
        assert result.questions_answered == 2
        assert result.score_percent == 100.0

    async def test_weak_topics_reflect_actually_wrong_answers(self, scripted):
        # Question n expects answer n. Getting Q2 wrong must surface topic2 only.
        answers = iter(["1", "999", "3", "done"])
        result = await run_quiz(
            "algebra", 600, ask=_from(answers), say=_silent, clock=FakeClock()
        )
        assert result.weak_topics == ["topic2"]
        assert result.score_percent == pytest.approx(66.7, abs=0.1)

    async def test_student_can_stop_early_with_done(self, scripted):
        answers = iter(["1", "done"])
        result = await run_quiz(
            "algebra", 600, ask=_from(answers), say=_silent, clock=FakeClock()
        )
        assert result.questions_answered == 1

    async def test_timer_stops_the_quiz(self, scripted):
        clock = FakeClock()

        async def slow_answer(_prompt, _left):
            clock.now += 40  # each answer eats 40 seconds
            return "1"

        result = await run_quiz(
            "algebra", 100, ask=slow_answer, say=_silent, clock=clock
        )
        # 40s, 80s answered; the third would end at 120s, past the 100s budget.
        assert result.questions_answered == 2
        assert result.time_used_seconds >= 100

    async def test_zero_budget_asks_nothing(self, scripted):
        async def never(_prompt, _left):
            raise AssertionError("should not have asked a question")

        result = await run_quiz("algebra", 0, ask=never, say=_silent, clock=FakeClock())
        assert result.questions_answered == 0


class TestFormatting:
    def test_result_is_rendered_not_dumped_as_json(self):
        result = QuizResult(
            score_percent=66.7,
            weak_topics=["mitosis"],
            questions_answered=3,
            time_used_seconds=95,
            verdict="Solid start; revise mitosis.",
        )
        out = format_result(result)
        assert "66.7%" in out
        assert "1m 35s" in out
        assert "mitosis" in out
        assert "{" not in out and '"' not in out

    def test_no_weak_topics_reads_naturally(self):
        out = format_result(
            QuizResult(
                score_percent=100,
                weak_topics=[],
                questions_answered=2,
                time_used_seconds=30,
                verdict="Clean sweep.",
            )
        )
        assert "none" in out


class TestGradingRouting:
    async def test_numeric_questions_are_graded_without_a_model_call(self, monkeypatch):
        async def explode(*_a, **_k):
            raise AssertionError("numeric grading must not call a model")

        monkeypatch.setattr(quiz_session.Runner, "run", explode)
        question = QuizQuestion(
            question="2+2?", expected_answer="4", topic="arithmetic", numeric=True
        )
        correct, _reason = await quiz_session._grade(question, "4")
        assert correct

    async def test_open_ended_questions_fall_through_to_the_model(self, monkeypatch):
        class FakeRun:
            final_output = GradeResult(correct=True, feedback="right idea")

        async def fake_run(*_a, **_k):
            return FakeRun()

        monkeypatch.setattr(quiz_session.Runner, "run", fake_run)
        question = QuizQuestion(
            question="Why does ice float?",
            expected_answer="lower density",
            topic="density",
            numeric=False,
        )
        correct, feedback = await quiz_session._grade(question, "it is less dense")
        assert correct and feedback == "right idea"


@pytest.mark.live
class TestAgainstRealModel:
    async def test_question_agent_returns_structured_output(self):
        from agents import Runner

        from study_helper.roles.quiz_agent import build_question_agent

        result = await Runner.run(build_question_agent(), "Ask the next question on: algebra.")
        question = result.final_output
        assert isinstance(question, QuizQuestion)
        assert question.question.strip()
        assert question.topic.strip()

    async def test_assessment_agent_returns_structured_output(self):
        from agents import Runner

        from study_helper.roles.quiz_agent import build_assessment_agent

        result = await Runner.run(
            build_assessment_agent(),
            "Questions the student got wrong:\n- [mitosis] What phase follows prophase? "
            "(expected metaphase, wrote telophase)",
        )
        assessment = result.final_output
        assert isinstance(assessment, QuizAssessment)
        assert assessment.verdict.strip()

    async def test_quiz_reached_through_the_orchestrator_handoff(self):
        """Written from the start this time, not after a silent failure."""
        from agents import Runner

        from study_helper.roles.orchestrator import build_orchestrator

        result = await Runner.run(build_orchestrator(), "Quiz me on algebra.")
        assert result.last_agent.name == "Quiz Master", (
            f"routed to {result.last_agent.name} instead of the Quiz Master"
        )
        assert result.final_output.strip()


class TestProviderSplit:
    """The quiz agents run on a different model from everything else."""

    def _rebuild(self):
        from study_helper import config

        config.build_model.cache_clear()
        config.build_structured_model.cache_clear()

    def test_falls_back_to_main_model_when_unset(self, monkeypatch):
        from study_helper import config

        monkeypatch.delenv("STUDY_HELPER_STRUCTURED_MODEL", raising=False)
        monkeypatch.setenv("STUDY_HELPER_API_KEY", "x")
        monkeypatch.setenv("STUDY_HELPER_MODEL", "main-model")
        self._rebuild()
        assert config.build_structured_model() is config.build_model()

    def test_structured_model_overrides_only_the_model(self, monkeypatch):
        from study_helper import config

        monkeypatch.setenv("STUDY_HELPER_API_KEY", "x")
        monkeypatch.setenv("STUDY_HELPER_BASE_URL", "https://main.example/v1")
        monkeypatch.setenv("STUDY_HELPER_MODEL", "main-model")
        monkeypatch.setenv("STUDY_HELPER_STRUCTURED_MODEL", "other-model")
        monkeypatch.delenv("STUDY_HELPER_STRUCTURED_BASE_URL", raising=False)
        monkeypatch.delenv("STUDY_HELPER_STRUCTURED_API_KEY", raising=False)
        self._rebuild()
        structured = config.build_structured_model()
        assert structured.model == "other-model"
        # Same provider, since only the model name was overridden.
        assert str(structured._client.base_url).startswith("https://main.example")

    def test_structured_model_can_use_a_different_provider(self, monkeypatch):
        from study_helper import config

        monkeypatch.setenv("STUDY_HELPER_API_KEY", "x")
        monkeypatch.setenv("STUDY_HELPER_BASE_URL", "https://main.example/v1")
        monkeypatch.setenv("STUDY_HELPER_MODEL", "main-model")
        monkeypatch.setenv("STUDY_HELPER_STRUCTURED_BASE_URL", "https://other.example/v1")
        monkeypatch.setenv("STUDY_HELPER_STRUCTURED_API_KEY", "y")
        monkeypatch.setenv("STUDY_HELPER_STRUCTURED_MODEL", "other-model")
        self._rebuild()
        assert str(config.build_structured_model()._client.base_url).startswith(
            "https://other.example"
        )
        assert config.build_model().model == "main-model"

    def test_tool_using_agents_stay_on_the_main_model(self, monkeypatch):
        """The whole point: tools keep the tool-capable provider."""
        from study_helper import config

        monkeypatch.setenv("STUDY_HELPER_API_KEY", "x")
        monkeypatch.setenv("STUDY_HELPER_MODEL", "main-model")
        monkeypatch.setenv("STUDY_HELPER_STRUCTURED_MODEL", "other-model")
        self._rebuild()

        from study_helper.roles.math_tutor import build_math_tutor
        from study_helper.roles.quiz_agent import build_question_agent, build_quiz_agent
        from study_helper.roles.science_tutor import build_science_tutor

        for build in (build_math_tutor, build_science_tutor, build_quiz_agent):
            assert build().model.model == "main-model", build.__name__
        assert build_question_agent().model.model == "other-model"


class TestSurvivesModelOutage:
    """The numbers are ours; only the prose depends on the model."""

    async def test_quiz_still_returns_a_result_when_assessment_fails(self, monkeypatch):
        questions = iter(
            QuizQuestion(question=f"Q{n}?", expected_answer=str(n), topic=f"t{n}", numeric=True)
            for n in range(1, 9)
        )

        async def fake_question(session, topic, asked):
            return next(questions)

        async def dead_assess(answers):
            raise RuntimeError("provider_overloaded")

        monkeypatch.setattr(quiz_session, "_next_question", fake_question)
        monkeypatch.setattr(quiz_session, "_assess", dead_assess)

        answers = iter(["1", "999", "done"])
        result = await run_quiz(
            "algebra", 600, ask=_from(answers), say=_silent, clock=FakeClock()
        )
        assert isinstance(result, QuizResult)
        assert result.questions_answered == 2
        assert result.score_percent == 50.0
        # Weak topics still correct, because they never came from the model.
        assert result.weak_topics == ["t2"]
        assert result.verdict

    def test_fallback_verdict_reports_the_real_numbers(self):
        answers = [_answer("a", True), _answer("b", False)]
        assessment = quiz_session.fallback_assessment(answers)
        assert assessment.weak_topics == ["b"]
        assert "1 of 2" in assessment.verdict

    def test_fallback_verdict_for_a_clean_sweep(self):
        assessment = quiz_session.fallback_assessment([_answer("a", True)])
        assert assessment.weak_topics == []
        assert "Nothing to revise" in assessment.verdict
