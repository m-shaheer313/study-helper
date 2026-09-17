"""Per-run state shared between agent callbacks and the run hooks.

The SDK hands the same RunContextWrapper to handoff callbacks and to RunHooks,
so `ctx.context` is the supported way to pass information between them. We use
it for one thing: the orchestrator states *why* it routed somewhere, and the
hooks attach that reason to the handoff event.

Everything here is optional. Runs started without a context still work; the
hooks simply report a handoff with no reason.
"""

from dataclasses import dataclass


@dataclass
class StudyContext:
    """Mutable, one per Runner.run call. Never share across runs."""

    routing_reason: str | None = None
