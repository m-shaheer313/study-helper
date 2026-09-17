"""The front desk: routes a question to the right tutor, teaches nothing."""

from agents import Agent, RunContextWrapper, handoff
from agents.extensions import handoff_filters
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from pydantic import BaseModel, Field

from study_helper.config import build_model
from study_helper.context import StudyContext
from study_helper.roles.math_tutor import build_math_tutor
from study_helper.roles.quiz_agent import build_quiz_agent
from study_helper.roles.science_tutor import build_science_tutor


class RoutingNote(BaseModel):
    """Why the front desk picked this tutor.

    This exists for two reasons. It makes the routing decision visible while
    you test, and it gives the handoff tool a non-empty JSON schema - some
    OpenAI-compatible providers (Groq) reject the SDK's default zero-argument
    handoff schema with "'required' present but 'properties' is missing".
    """

    subject: str = Field(description="The subject you are routing to.")
    reason: str = Field(description="One short clause on why this tutor fits.")


async def _announce(ctx: RunContextWrapper, note: RoutingNote) -> None:
    """Record why the front desk routed where it did. Never prints.

    RoutingNote also has to exist for a second reason: it gives the handoff
    tool a non-empty JSON schema, without which Groq rejects the call.

    The reason is stashed on the run context so the hooks can attach it to the
    handoff event. Displaying it belongs to whoever is consuming events - a
    terminal or a UI - not to this layer.
    """
    context = getattr(ctx, "context", None)
    if isinstance(context, StudyContext):
        context.routing_reason = f"{note.subject}: {note.reason}"

ORCHESTRATOR_INSTRUCTIONS = f"""{RECOMMENDED_PROMPT_PREFIX}

You are the front desk of an exam-prep study helper. You do not teach.
Your only job is to route the student's question to the right specialist.

ROUTING
- Math (arithmetic, algebra, geometry, trigonometry, calculus, probability,
  statistics, discrete math, linear algebra) -> hand off to Math Tutor.
- Science (physics, chemistry, biology, earth science) -> hand off to
  Science Tutor.
- Asking to be TESTED rather than taught ("quiz me", "test me", "ask me
  questions", "give me practice questions") -> hand off to Quiz Master,
  whatever the subject.
- If a question touches both, route on what the student is actually being
  tested on: prefer Science when it asks why a physical phenomenon happens,
  prefer Math when it asks how to solve or manipulate an equation.

OFF TOPIC
- If the question is neither math nor science, do NOT hand off. Reply yourself
  in one friendly line saying you only cover math and science, and invite such
  a question. Do not answer the off-topic question itself.

NEVER
- Never explain, teach, or solve anything yourself. If it is math or science,
  hand off - the specialist does all the explaining.
- Never announce or describe the handoff.
"""


def build_orchestrator() -> Agent:
    return Agent(
        name="Study Helper Front Desk",
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        model=build_model(),
        handoffs=[
            handoff(
                agent=build_math_tutor(),
                input_type=RoutingNote,
                on_handoff=_announce,
                # Without this the tutor inherits the completed transfer_to_*
                # call and reads it as "tool phase over", then answers from
                # memory. Measured: 0 calculate calls with the transfer in
                # history, 7 with it stripped.
                input_filter=handoff_filters.remove_all_tools,
            ),
            handoff(
                agent=build_science_tutor(),
                input_type=RoutingNote,
                on_handoff=_announce,
                input_filter=handoff_filters.remove_all_tools,
            ),
            handoff(
                agent=build_quiz_agent(),
                input_type=RoutingNote,
                on_handoff=_announce,
                input_filter=handoff_filters.remove_all_tools,
            ),
        ],
    )
