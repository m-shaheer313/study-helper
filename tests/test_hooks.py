"""Tests for run observability.

Most of this runs offline by driving the hook methods directly - they are
ordinary async methods, so no model is needed to check what they emit. The
live test confirms the SDK calls them in the order we expect for a real
handoff-plus-tool run.
"""

import pytest

from study_helper.hooks import (
    MAX_DETAIL,
    EventEmittingHooks,
    EventKind,
    RunEvent,
    format_event,
    record_events,
)


class Named:
    """Stands in for an Agent or a tool - both are identified by .name."""

    def __init__(self, name: str) -> None:
        self.name = name


MATH = Named("Math Tutor")
DESK = Named("Study Helper Front Desk")
CALC = Named("calculate")


class TestEmitsStructuredEvents:
    async def test_agent_start_and_end(self):
        events, hooks = record_events()
        await hooks.on_agent_start(None, MATH)
        await hooks.on_agent_end(None, MATH, "the answer is 144")

        assert [e.kind for e in events] == [EventKind.AGENT_START, EventKind.AGENT_END]
        assert all(e.agent == "Math Tutor" for e in events)
        assert events[1].detail == "the answer is 144"
        assert events[1].duration_ms is not None

    async def test_handoff_records_both_sides(self):
        events, hooks = record_events()
        await hooks.on_handoff(None, DESK, MATH)

        (event,) = events
        assert event.kind == EventKind.HANDOFF
        assert event.agent == "Study Helper Front Desk"
        assert event.target == "Math Tutor"

    async def test_tool_start_and_end_carry_the_tool_name(self):
        events, hooks = record_events()
        await hooks.on_tool_start(None, MATH, CALC)
        await hooks.on_tool_end(None, MATH, CALC, 144.0)

        assert [e.kind for e in events] == [EventKind.TOOL_START, EventKind.TOOL_END]
        assert all(e.tool == "calculate" for e in events)
        assert events[1].detail == "144.0"
        assert events[1].duration_ms is not None

    async def test_events_are_immutable(self):
        events, hooks = record_events()
        await hooks.on_agent_start(None, MATH)
        with pytest.raises(Exception):
            events[0].agent = "tampered"

    async def test_long_tool_results_are_truncated(self):
        events, hooks = record_events()
        await hooks.on_tool_start(None, MATH, CALC)
        await hooks.on_tool_end(None, MATH, CALC, "x" * 500)
        assert len(events[-1].detail) <= MAX_DETAIL

    async def test_multiline_detail_is_collapsed_to_one_line(self):
        events, hooks = record_events()
        await hooks.on_tool_end(None, MATH, CALC, "line one\nline two")
        assert "\n" not in events[-1].detail


class TestCallbackIsolation:
    async def test_a_broken_callback_cannot_break_the_run(self):
        """Observability must never become a new failure mode."""

        def explode(_event):
            raise RuntimeError("consumer bug")

        hooks = EventEmittingHooks(explode)
        # None of these may raise.
        await hooks.on_agent_start(None, MATH)
        await hooks.on_handoff(None, DESK, MATH)
        await hooks.on_tool_start(None, MATH, CALC)
        await hooks.on_tool_end(None, MATH, CALC, 1)
        await hooks.on_agent_end(None, MATH, "done")


class TestFormatting:
    def test_handoff_reads_naturally(self):
        line = format_event(
            RunEvent(EventKind.HANDOFF, "Study Helper Front Desk", 0.0, target="Science Tutor")
        )
        assert line == "   [Study Helper Front Desk] -> routing to Science Tutor"

    def test_known_tools_get_a_friendly_verb(self):
        assert "calculating" in format_event(
            RunEvent(EventKind.TOOL_START, "Math Tutor", 0.0, tool="calculate")
        )
        assert "searching the web" in format_event(
            RunEvent(EventKind.TOOL_START, "Science Tutor", 0.0, tool="web_search")
        )

    def test_unknown_tools_still_render(self):
        line = format_event(RunEvent(EventKind.TOOL_START, "A", 0.0, tool="mystery"))
        assert "mystery" in line

    def test_agent_lifecycle_is_hidden_from_the_terminal(self):
        # A UI may want these; a terminal does not.
        assert format_event(RunEvent(EventKind.AGENT_START, "Math Tutor", 0.0)) is None
        assert format_event(RunEvent(EventKind.AGENT_END, "Math Tutor", 0.0)) is None


class TestPurelyObservational:
    def test_hooks_are_opt_in_per_call(self):
        """Nothing about an Agent changes, so existing behaviour cannot shift."""
        import inspect

        from agents import Runner

        from study_helper.roles.math_tutor import build_math_tutor

        assert "hooks" in inspect.signature(Runner.run).parameters
        # Agents carry no hooks of their own; observability is caller-supplied.
        assert build_math_tutor().hooks is None


@pytest.mark.live
class TestRealRunSequence:
    async def test_handoff_then_tool_fires_in_order(self):
        """A cleaner statement of composition than inspecting new_items."""
        from agents import Runner

        from study_helper.roles.orchestrator import build_orchestrator

        events, hooks = record_events()
        await Runner.run(build_orchestrator(), "What is 12 * 12?", hooks=hooks)

        kinds = [e.kind for e in events]
        assert kinds[0] == EventKind.AGENT_START
        assert events[0].agent == "Study Helper Front Desk"

        # Handoffs go through on_handoff only - they never fire on_tool_start.
        handoff = next(e for e in events if e.kind == EventKind.HANDOFF)
        assert handoff.target == "Math Tutor"

        tool_start = next(e for e in events if e.kind == EventKind.TOOL_START)
        assert tool_start.tool == "calculate"
        assert tool_start.agent == "Math Tutor"

        # Ordering is the thing new_items cannot state this directly:
        # the tutor must be handed the work before it starts calculating.
        assert kinds.index(EventKind.HANDOFF) < kinds.index(EventKind.TOOL_START)
        assert kinds.index(EventKind.TOOL_START) < kinds.index(EventKind.TOOL_END)
        assert kinds[-1] == EventKind.AGENT_END
        assert events[-1].agent == "Math Tutor"

    async def test_no_handoff_tool_call_is_reported_as_a_tool(self):
        from agents import Runner

        from study_helper.roles.orchestrator import build_orchestrator

        from agents.exceptions import OutputGuardrailTripwireTriggered

        events, hooks = record_events()
        try:
            await Runner.run(build_orchestrator(), "Why does ice float?", hooks=hooks)
        except OutputGuardrailTripwireTriggered:
            # A leaked-answer tripwire is valid behaviour here and aborts the
            # run, but the events fired before it are what we are asserting on.
            pass

        assert events, "no events were emitted at all"
        tools = [e.tool for e in events if e.kind == EventKind.TOOL_START]
        assert not any(t.startswith("transfer_to") for t in tools), tools


class TestRoutingReason:
    """The orchestrator's reason for routing rides on the handoff event."""

    async def test_reason_is_attached_when_a_context_is_present(self):
        from agents.run_context import RunContextWrapper

        from study_helper.context import StudyContext
        from study_helper.roles.orchestrator import RoutingNote, _announce

        ctx = RunContextWrapper(StudyContext())
        await _announce(ctx, RoutingNote(subject="Algebra", reason="asks to factor"))

        events, hooks = record_events()
        await hooks.on_handoff(ctx, DESK, MATH)
        assert events[0].detail == "Algebra: asks to factor"

    async def test_handoff_without_a_context_still_emits(self):
        """Runs started without a StudyContext must not break."""
        events, hooks = record_events()
        await hooks.on_handoff(None, DESK, MATH)
        assert events[0].kind == EventKind.HANDOFF
        assert events[0].detail is None

    async def test_announce_is_silent_without_a_study_context(self, capsys):
        from agents.run_context import RunContextWrapper

        from study_helper.roles.orchestrator import RoutingNote, _announce

        # A foreign context object must be left alone, not crashed on.
        await _announce(RunContextWrapper(object()), RoutingNote(subject="A", reason="b"))
        assert capsys.readouterr().out == ""

    def test_formatted_handoff_includes_the_reason(self):
        line = format_event(
            RunEvent(
                EventKind.HANDOFF,
                "Study Helper Front Desk",
                0.0,
                target="Math Tutor",
                detail="Arithmetic: a multiplication question",
            )
        )
        assert line == (
            "   [Study Helper Front Desk] -> routing to Math Tutor "
            "(Arithmetic: a multiplication question)"
        )

    def test_formatted_handoff_without_a_reason_is_unchanged(self):
        line = format_event(
            RunEvent(EventKind.HANDOFF, "Front Desk", 0.0, target="Math Tutor")
        )
        assert line == "   [Front Desk] -> routing to Math Tutor"
