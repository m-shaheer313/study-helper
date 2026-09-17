"""Chainlit web UI for study-helper.

A second interface over exactly the same system as the terminal. It builds the
same orchestrator, runs the same tools and guardrails, and consumes the same
RunEvent stream. No agent logic lives here.

Run it with:  uv run chainlit run chainlit_app.py

Two integration details worth knowing:

1. EventEmittingHooks takes a *synchronous* callback, but every Chainlit render
   call is async. So the callback is queue.put_nowait - sync and non-blocking -
   and a separate task drains the queue and renders. The queue is FIFO, so
   steps appear in the order they happened, and hooks.py needs no changes.

2. State lives in cl.user_session, which is per browser connection. Module
   level state would leak one student's conversation into another's.
"""

import asyncio
import contextlib

import chainlit as cl
from agents import Runner, SQLiteSession
from agents.exceptions import OutputGuardrailTripwireTriggered

from study_helper.cli import to_ascii
from study_helper.context import StudyContext
from study_helper.hooks import EventEmittingHooks, EventKind, RunEvent
from study_helper.quiz_session import format_result, run_quiz
from study_helper.roles.orchestrator import build_orchestrator
from study_helper.turns import atomic_turn

WELCOME = (
    "**Study Helper** - ask a math or science question, "
    "or type `quiz` to start a timed quiz."
)

_TOOL_LABELS = {
    "calculate": "Calculating",
    "web_search": "Searching the web",
}


async def _render(event: RunEvent, open_steps: dict[str, cl.Step]) -> None:
    """Turn one RunEvent into Chainlit UI. Never raises."""
    if event.kind is EventKind.HANDOFF:
        why = f" - {event.detail}" if event.detail else ""
        await cl.Message(
            content=f"Routing to **{event.target}**{why}", author="Front Desk"
        ).send()

    elif event.kind is EventKind.TOOL_START:
        label = _TOOL_LABELS.get(event.tool or "", f"Running {event.tool}")
        step = cl.Step(name=f"{label}", type="tool")
        step.input = event.tool
        await step.__aenter__()
        open_steps[event.tool or ""] = step

    elif event.kind is EventKind.TOOL_END:
        step = open_steps.pop(event.tool or "", None)
        if step is not None:
            took = f" ({event.duration_ms}ms)" if event.duration_ms is not None else ""
            step.output = f"{event.detail}{took}"
            await step.__aexit__(None, None, None)


async def _consume(queue: asyncio.Queue, open_steps: dict[str, cl.Step]) -> None:
    """Drain events and render them until cancelled."""
    while True:
        event = await queue.get()
        try:
            await _render(event, open_steps)
        except Exception:
            # A rendering failure must not take down the conversation.
            pass
        finally:
            queue.task_done()


@contextlib.asynccontextmanager
async def _live_status():
    """Render run events as they happen, closing any steps left open."""
    queue: asyncio.Queue[RunEvent] = asyncio.Queue()
    open_steps: dict[str, cl.Step] = {}
    consumer = asyncio.create_task(_consume(queue, open_steps))
    try:
        yield EventEmittingHooks(queue.put_nowait)
        await queue.join()
    finally:
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer
        # An aborted run can leave a tool step open; close it rather than
        # leaving a spinner running forever in the browser.
        for step in open_steps.values():
            with contextlib.suppress(Exception):
                await step.__aexit__(None, None, None)


@cl.on_chat_start
async def start() -> None:
    # Per browser session, so two students never share history.
    cl.user_session.set("orchestrator", build_orchestrator())
    cl.user_session.set("session", SQLiteSession("chainlit"))
    await cl.Message(content=WELCOME).send()


async def _answer(question: str) -> None:
    orchestrator = cl.user_session.get("orchestrator")
    session = cl.user_session.get("session")

    async with _live_status() as hooks:
        try:
            # Chainlit spawns a task per message, so a second question sent
            # while this one is running would share the session and interleave.
            # atomic_turn serialises them and discards a failed turn.
            async with atomic_turn(session):
                try:
                    result = await Runner.run(
                        orchestrator,
                        question,
                        session=session,
                        hooks=hooks,
                        context=StudyContext(),
                    )
                    answer = to_ascii(result.final_output)
                    author = result.last_agent.name
                except OutputGuardrailTripwireTriggered as tripped:
                    # This turn succeeded - the reply just needed cleaning - so
                    # it is handled inside and never rolled back.
                    report = tripped.guardrail_result.output.output_info
                    answer = to_ascii(report.cleaned)
                    author = tripped.guardrail_result.agent.name
                    await session.add_items([{"role": "assistant", "content": answer}])
        except Exception as err:
            # Never show a traceback to a student.
            await cl.Message(
                content=f"Something went wrong talking to the model ({type(err).__name__}). "
                "Please try again.",
                author="Study Helper",
            ).send()
            return

    await cl.Message(content=answer, author=author).send()


async def _quiz() -> None:
    """Quiz mode. Same loop as the terminal, different ask/say."""
    topic_reply = await cl.AskUserMessage(
        content="What should I quiz you on?", timeout=120
    ).send()
    if not topic_reply:
        await cl.Message(content="No topic - no quiz.").send()
        return
    topic = topic_reply["output"].strip()

    minutes_reply = await cl.AskUserMessage(
        content="How many minutes do you want? (e.g. 5)", timeout=120
    ).send()
    minutes = 5.0
    if minutes_reply:
        try:
            parsed = float(minutes_reply["output"].strip())
            if parsed > 0:
                minutes = parsed
        except ValueError:
            await cl.Message(content="That wasn't a number of minutes - using 5.").send()

    await cl.Message(
        content=f"**{minutes:g} minutes on {topic}.** Answer each question, "
        "or say `done` to stop early. If you run out of time I'll stop for you."
    ).send()

    async def ask(prompt: str, seconds_left: float) -> str | None:
        # The real timer: no answer within the remaining budget ends the quiz,
        # even if the student has walked away.
        reply = await cl.AskUserMessage(
            content=prompt.strip(), timeout=max(1, int(seconds_left))
        ).send()
        return None if reply is None else reply["output"]

    async def say(message: str) -> None:
        text = to_ascii(message).strip()
        if text:
            await cl.Message(content=text, author="Quiz Master").send()

    try:
        result = await run_quiz(topic, minutes * 60, ask=ask, say=say)
    except Exception as err:
        await cl.Message(
            content=f"The quiz stopped unexpectedly ({type(err).__name__}). "
            "Type `quiz` to start another."
        ).send()
        return

    await cl.Message(content=f"```\n{to_ascii(format_result(result))}\n```").send()


@cl.on_message
async def message(incoming: cl.Message) -> None:
    text = incoming.content.strip()
    if not text:
        return
    if text.lower() == "quiz":
        await _quiz()
        return
    await _answer(text)


@cl.on_chat_end
async def end() -> None:
    session = cl.user_session.get("session")
    if session is not None:
        with contextlib.suppress(Exception):
            session.close()
