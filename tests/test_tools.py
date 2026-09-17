"""Tests for the calculator tool.

The evaluator tests run offline. The test that proves the Math Tutor actually
calls the tool is marked `live` because it needs a real model.
"""

import pytest

from study_helper.tools import CalculationError, calculate, evaluate_expression


class TestArithmetic:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            # The failure that motivated this tool.
            ("1234 * 5678", 7006652),
            ("34 * (58 + 9)", 2278),
            ("2 + 3 * 4", 14),
            ("(2 + 3) * 4", 20),
            ("10 / 4", 2.5),
            ("10 // 4", 2),
            ("10 % 4", 2),
            ("2 ** 10", 1024),
            ("-5 + 3", -2),
            ("-(4 - 9)", 5),
            ("1.5 * 2", 3.0),
        ],
    )
    def test_evaluates_correctly(self, expression, expected):
        assert evaluate_expression(expression) == expected

    def test_always_returns_a_float(self):
        assert isinstance(evaluate_expression("2 + 2"), float)


class TestRejectsBadInput:
    def test_division_by_zero(self):
        with pytest.raises(CalculationError, match="division by zero"):
            evaluate_expression("1 / 0")

    @pytest.mark.parametrize("expression", ["5 // 0", "5 % 0"])
    def test_integer_division_by_zero(self, expression):
        with pytest.raises(CalculationError, match="division by zero"):
            evaluate_expression(expression)

    @pytest.mark.parametrize("expression", ["2 +", "((3)", "", "* 5"])
    def test_syntax_errors_are_reported_clearly(self, expression):
        with pytest.raises(CalculationError, match="could not parse"):
            evaluate_expression(expression)

    def test_huge_exponent_is_refused_rather_than_hanging(self):
        with pytest.raises(CalculationError, match="too large"):
            evaluate_expression("2 ** 10000000")


class TestRefusesCodeExecution:
    """The whole point of walking the AST instead of calling eval()."""

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('echo pwned')",
            "open('secrets.txt').read()",
            "print(1)",
            "x + 1",
            "[1, 2, 3]",
            "lambda: 1",
            "1 if True else 2",
            "'a' * 3",
        ],
    )
    def test_non_arithmetic_is_rejected(self, expression):
        with pytest.raises(CalculationError):
            evaluate_expression(expression)

    def test_booleans_are_not_numbers(self):
        with pytest.raises(CalculationError, match="booleans"):
            evaluate_expression("True + 1")


class TestToolWiring:
    def test_tool_is_registered_with_a_description(self):
        assert calculate.name == "calculate"
        assert calculate.description, "the docstring should become the tool description"

    def test_math_tutor_has_the_calculator(self):
        from study_helper.roles.math_tutor import build_math_tutor

        assert "calculate" in [t.name for t in build_math_tutor().tools]

    def test_science_tutor_does_not(self):
        from study_helper.roles.science_tutor import build_science_tutor

        assert "calculate" not in [t.name for t in build_science_tutor().tools]


@pytest.mark.live
class TestToolIsActuallyUsed:
    """Checking the final answer alone is worthless - a right answer could be
    a lucky guess. These inspect the run's item trace instead."""

    async def test_math_tutor_calls_calculate(self):
        from agents import Runner
        from agents.items import ToolCallItem, ToolCallOutputItem

        from study_helper.roles.math_tutor import build_math_tutor

        # Guardrails are tested separately; a tripwire here would abort the run
        # and hide new_items, so isolate the behaviour under test.
        tutor = build_math_tutor().clone(output_guardrails=[])
        result = await Runner.run(tutor, "What is 1234 * 5678?")

        calls = [i for i in result.new_items if isinstance(i, ToolCallItem)]
        assert calls, "Math Tutor answered without calling calculate"

        outputs = [
            float(i.output) for i in result.new_items if isinstance(i, ToolCallOutputItem)
        ]
        assert 7006652 in outputs, f"tool never returned the product: {outputs}"

        # The model can ignore its own tool result, so verify it reported it.
        assert "7006652" in result.final_output, "model overrode its own tool result"

    async def test_decomposes_into_separate_calls(self):
        from agents import Runner
        from agents.items import ToolCallItem

        from study_helper.roles.math_tutor import build_math_tutor

        tutor = build_math_tutor().clone(output_guardrails=[])
        result = await Runner.run(tutor, "What is 1234 * 5678?")

        calls = [i for i in result.new_items if isinstance(i, ToolCallItem)]
        # Partial products plus running sums, not one call on the whole thing.
        assert len(calls) > 1, f"collapsed into {len(calls)} call(s); steps hide the working"

    async def test_tool_survives_the_handoff(self):
        # Regression: the tutor used to inherit the completed transfer_to_*
        # call and treat the tool phase as finished, making zero calculate
        # calls when reached through the orchestrator.
        from agents import Runner
        from agents.items import ToolCallItem

        from study_helper.roles.orchestrator import build_orchestrator

        result = await Runner.run(build_orchestrator(), "What is 1234 * 5678?")

        names = [
            str(getattr(i.raw_item, "name", None) or i.raw_item.get("name", ""))
            for i in result.new_items
            if isinstance(i, ToolCallItem)
        ]
        assert any("calculate" in n for n in names), (
            f"no calculate call after handoff; tool calls were {names}"
        )
        assert "7006652" in result.final_output
