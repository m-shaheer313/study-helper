"""Observability: turn a run's internal steps into a stream of structured events.

We kept debugging tool and handoff behaviour after the fact by picking through
result.new_items in tests. Hooks make the same information available live, and
in order - which is the part new_items cannot express clearly.

The hooks emit data, never text. cli.py formats events for a terminal; a UI can
map the same events to its own widgets without parsing our strings.

RunHooks rather than AgentHooks on purpose: RunHooks attaches to Runner.run and
follows the conversation across handoffs, and its on_handoff reports both sides.
AgentHooks would need one object per agent and only tells the receiving agent
who sent it.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from agents import RunHooks

MAX_DETAIL = 80


class EventKind(StrEnum):
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    HANDOFF = "handoff"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"


@dataclass(frozen=True, slots=True)
class RunEvent:
    """One observable step. Frozen so a consumer cannot mutate shared state."""

    kind: EventKind
    agent: str
    at: float
    tool: str | None = None
    target: str | None = None
    detail: str | None = None
    duration_ms: int | None = None


Emit = Callable[[RunEvent], None]


def _name(thing: Any) -> str:
    return str(getattr(thing, "name", None) or thing.__class__.__name__)


def _preview(value: Any) -> str | None:
    """A short single-line summary, so a big tool result cannot flood a UI."""
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    return text if len(text) <= MAX_DETAIL else text[: MAX_DETAIL - 1] + "…"


class EventEmittingHooks(RunHooks):
    """Emits a RunEvent for each lifecycle step. Purely observational."""

    def __init__(self, emit: Emit) -> None:
        self._emit = emit
        self._started: dict[str, float] = {}

    def _send(self, event: RunEvent) -> None:
        # A hook that raises would abort the run, turning observability into a
        # new failure mode. A broken callback must never break a quiz.
        try:
            self._emit(event)
        except Exception:
            pass

    def _begin(self, key: str) -> None:
        self._started[key] = time.monotonic()

    def _elapsed_ms(self, key: str) -> int | None:
        start = self._started.pop(key, None)
        return None if start is None else int((time.monotonic() - start) * 1000)

    async def on_agent_start(self, context, agent) -> None:
        name = _name(agent)
        self._begin(f"agent:{name}")
        self._send(RunEvent(EventKind.AGENT_START, name, time.time()))

    async def on_agent_end(self, context, agent, output) -> None:
        name = _name(agent)
        self._send(
            RunEvent(
                EventKind.AGENT_END,
                name,
                time.time(),
                detail=_preview(output),
                duration_ms=self._elapsed_ms(f"agent:{name}"),
            )
        )

    async def on_handoff(self, context, from_agent, to_agent) -> None:
        # The orchestrator's handoff callback stashes its reason on the run
        # context before this fires. Absent when the run had no StudyContext.
        state = getattr(context, "context", None)
        self._send(
            RunEvent(
                EventKind.HANDOFF,
                _name(from_agent),
                time.time(),
                target=_name(to_agent),
                detail=_preview(getattr(state, "routing_reason", None)),
            )
        )

    async def on_tool_start(self, context, agent, tool) -> None:
        tool_name = _name(tool)
        self._begin(f"tool:{tool_name}")
        self._send(RunEvent(EventKind.TOOL_START, _name(agent), time.time(), tool=tool_name))

    async def on_tool_end(self, context, agent, tool, result) -> None:
        tool_name = _name(tool)
        self._send(
            RunEvent(
                EventKind.TOOL_END,
                _name(agent),
                time.time(),
                tool=tool_name,
                detail=_preview(result),
                duration_ms=self._elapsed_ms(f"tool:{tool_name}"),
            )
        )


def record_events() -> tuple[list[RunEvent], EventEmittingHooks]:
    """Hooks that append to a list. For tests and debugging."""
    captured: list[RunEvent] = []
    return captured, EventEmittingHooks(captured.append)


_TOOL_VERBS = {
    "calculate": "calculating",
    "web_search": "searching the web",
}


def format_event(event: RunEvent) -> str | None:
    """Render an event for a terminal. None means "not worth showing"."""
    match event.kind:
        case EventKind.HANDOFF:
            why = f" ({event.detail})" if event.detail else ""
            return f"   [{event.agent}] -> routing to {event.target}{why}"
        case EventKind.TOOL_START:
            verb = _TOOL_VERBS.get(event.tool or "", f"running {event.tool}")
            return f"   [{event.agent}] -> {verb}..."
        case EventKind.TOOL_END:
            took = f" ({event.duration_ms}ms)" if event.duration_ms is not None else ""
            return f"   [{event.agent}] <- {event.tool}{took}: {event.detail}"
        case _:
            # Agent start/end are noise in a terminal; a UI may still want them.
            return None
