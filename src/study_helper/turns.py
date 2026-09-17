"""Make one conversational turn atomic.

Two failure modes put a stale question back in front of the model, and both
show up to the user the same way: you ask something new and get an answer to
something you asked earlier.

1. A failed run leaves its question behind. The SDK writes the input items to
   the session with an empty output list *before* calling the model
   (agents/run.py). If that call then fails, the question stays in the history
   with no reply, and the next turn re-sends it - so the model answers the old
   question instead of the new one.

2. Overlapping runs interleave. Chainlit spawns a task per incoming message
   (chainlit/socket.py), so sending a second message while the first is still
   running gives two runs one session. Measured: the transcript came out as
   user A, user B, assistant B, assistant A, and the second answer re-answered
   the first question because that question was sitting there unanswered.

`atomic_turn` closes both: one turn at a time per session, and a turn that
fails leaves nothing behind.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from weakref import WeakKeyDictionary

from agents.memory import Session

# Keyed weakly so a closed browser session's lock is collected with it.
_locks: WeakKeyDictionary[Session, asyncio.Lock] = WeakKeyDictionary()


def lock_for(session: Session) -> asyncio.Lock:
    """The lock guarding one session. One per session, created on demand."""
    lock = _locks.get(session)
    if lock is None:
        lock = asyncio.Lock()
        _locks[session] = lock
    return lock


async def rollback_to(session: Session, item_count: int) -> None:
    """Drop everything added to the session since it held `item_count` items."""
    for _ in range(len(await session.get_items()) - item_count):
        await session.pop_item()


@asynccontextmanager
async def atomic_turn(session: Session) -> AsyncIterator[None]:
    """Run one turn with the session to itself, discarding it if it fails.

    A turn that raises is rolled back, so re-asking is a clean retry rather
    than a question the model sees twice.
    """
    async with lock_for(session):
        before = len(await session.get_items())
        try:
            yield
        except BaseException:
            await rollback_to(session, before)
            raise
