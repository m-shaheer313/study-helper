"""Tests for the leaked-answer output guardrail.

The detector tests are pure and run offline. The one test that proves the SDK
actually invokes the guardrail is marked `live` because it needs a real model.
"""

import pytest

from study_helper.guardrails import PLACEHOLDER, no_leaked_answer, repair_check_yourself

# The real failure this guardrail exists for, captured verbatim from a run.
REAL_LEAK = """**Check yourself**
Compute 4321 * 8765 and see if you get 37864,? (fill in the result)."""

CLEAN = """**Check yourself**
Compute 4321 * 8765 using the same method. (Give it a try.)"""


class TestDetectsLeaks:
    def test_catches_the_real_world_failure(self):
        report = repair_check_yourself(REAL_LEAK)
        assert report.leaked
        assert "37864" not in report.cleaned
        assert PLACEHOLDER in report.cleaned
        # The practice question itself must survive.
        assert "4321 * 8765" in report.cleaned

    def test_catches_heading_and_question_on_one_line(self):
        # Regression: an earlier heading regex ate the rest of the line,
        # leaving nothing to inspect, so this shape went undetected.
        report = repair_check_yourself(
            "**Check yourself**  Compute 4321 * 8765 and see if you get 37864."
        )
        assert report.leaked
        assert "37864" not in report.cleaned

    def test_catches_digits_after_the_question_mark(self):
        report = repair_check_yourself("**Check yourself**\nWhat is 12 * 12? Answer: 144")
        assert report.leaked
        assert "144" not in report.cleaned

    @pytest.mark.parametrize(
        "leak",
        [
            "the answer is 42",
            "you should get 42",
            "it should be 42",
            "which equals 42",
        ],
    )
    def test_catches_each_leak_phrase(self, leak):
        report = repair_check_yourself(f"**Check yourself**\nWhat is 6 * 7 - {leak}.")
        assert report.leaked
        assert "42" not in report.cleaned

    def test_repair_leaves_brackets_balanced(self):
        # Truncating mid-bracket used to emit "(Try it. (Solve this yourself...)".
        report = repair_check_yourself(
            "**Check yourself**\nWhat is 6 * 7? (Try it and see if you get 42.)"
        )
        assert report.leaked
        assert "42" not in report.cleaned
        assert report.cleaned.count("(") == report.cleaned.count(")")

    def test_reports_evidence_of_what_was_removed(self):
        report = repair_check_yourself(REAL_LEAK)
        assert "37864" in report.evidence


class TestLeavesGoodOutputAlone:
    def test_clean_section_does_not_trip(self):
        assert not repair_check_yourself(CLEAN).leaked

    def test_digits_inside_the_question_are_not_a_leak(self):
        report = repair_check_yourself(
            "**Check yourself**\nIf a 200 g block of ice occupies 215 cm3, will it float?"
        )
        assert not report.leaked

    @pytest.mark.parametrize(
        "tail",
        [
            "(Try it and see if you get the same result as a calculator.)",
            "(Give it a try.)",
            "Use the same steps and see what you obtain.",
        ],
    )
    def test_leak_phrasing_without_a_number_is_not_a_leak(self, tail):
        # Regression: "see if you get the same result as a calculator" gives
        # nothing away, but phrase-matching alone flagged it.
        report = repair_check_yourself(f"**Check yourself**\nWhat is 4321 * 8765? {tail}")
        assert not report.leaked

    def test_answer_without_a_check_yourself_section_is_ignored(self):
        # The guardrail must not police the rest of the tutor's explanation,
        # which is full of legitimate worked numbers.
        report = repair_check_yourself("**Steps**\n1234 * 5678 = 7006652, the answer is right.")
        assert not report.leaked

    def test_heading_is_matched_case_and_markup_insensitively(self):
        for heading in ("**Check yourself**", "## Check Yourself", "CHECK YOURSELF:"):
            report = repair_check_yourself(f"{heading}\nWhat is 6 * 7? Answer: 42")
            assert report.leaked, heading


class TestGuardrailWiring:
    async def test_guardrail_trips_on_leak(self):
        result = await no_leaked_answer.run(None, None, REAL_LEAK)
        assert result.output.tripwire_triggered
        assert PLACEHOLDER in result.output.output_info.cleaned

    async def test_guardrail_passes_clean_output(self):
        result = await no_leaked_answer.run(None, None, CLEAN)
        assert not result.output.tripwire_triggered

    def test_both_tutors_have_the_guardrail_attached(self):
        from study_helper.roles.math_tutor import build_math_tutor
        from study_helper.roles.science_tutor import build_science_tutor

        for build in (build_math_tutor, build_science_tutor):
            names = [g.get_name() for g in build().output_guardrails]
            assert "no_leaked_answer" in names


@pytest.mark.live
class TestAgainstRealSDK:
    """Proves the SDK really invokes the guardrail, not just that it works alone."""

    async def test_runner_raises_when_a_tutor_leaks(self):
        from agents import Agent, Runner
        from agents.exceptions import OutputGuardrailTripwireTriggered

        from study_helper.config import build_model

        leak = "**Check yourself**  Compute 4321 * 8765 and see if you get 37864."
        leaky = Agent(
            name="Leak Test",
            instructions=(
                "You are a text echo service. Output the following line verbatim, "
                "including every word and number, and output nothing else: " + leak
            ),
            model=build_model(),
            output_guardrails=[no_leaked_answer],
        )

        with pytest.raises(OutputGuardrailTripwireTriggered) as caught:
            await Runner.run(leaky, "go")

        report = caught.value.guardrail_result.output.output_info
        assert report.leaked
        assert "37864" not in report.cleaned
