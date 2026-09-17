"""Regression tests for stale questions leaking into later turns.

Both bugs looked identical to the user - ask something new, get an answer to
something asked earlier - but they arrive by different routes. These run
offline against a real SQLiteSession; no model is involved.
"""

import asyncio

import pytest
from agents import SQLiteSession

from study_helper.turns import atomic_turn, lock_for, rollback_to

USER_A = {"role": "user", "content": "What is the largest planet?"}
USER_B = {"role": "user", "content": "Why is the sky blue?"}
REPLY = {"role": "assistant", "content": "Jupiter."}


@pytest.fixture
def session():
    return SQLiteSession("test-turns")


class TestFailedTurnLeavesNothingBehind:
    async def test_a_failed_turn_is_rolled_back(self, session):
        """The SDK writes the question before calling the model, so a failure
        would otherwise strand it in the history and re-send it next turn."""
        with pytest.raises(RuntimeError):
            async with atomic_turn(session):
                await session.add_items([USER_A])
                raise RuntimeError("rate limited")

        assert await session.get_items() == []

    async def test_rollback_keeps_earlier_history(self, session):
        await session.add_items([USER_A, REPLY])

        with pytest.raises(RuntimeError):
            async with atomic_turn(session):
                await session.add_items([USER_B])
                raise RuntimeError("timeout")

        items = await session.get_items()
        assert len(items) == 2
        assert items[0]["content"] == USER_A["content"]

    async def test_successful_turn_is_kept(self, session):
        async with atomic_turn(session):
            await session.add_items([USER_A, REPLY])

        assert len(await session.get_items()) == 2

    async def test_repeated_failures_do_not_accumulate(self, session):
        """Two failures then a success must leave exactly one clean turn -
        this is the reported bug, where two dead questions got answered."""
        for _ in range(2):
            with pytest.raises(RuntimeError):
                async with atomic_turn(session):
                    await session.add_items([USER_A])
                    raise RuntimeError("boom")

        async with atomic_turn(session):
            await session.add_items([USER_B, REPLY])

        items = await session.get_items()
        assert len(items) == 2
        assert items[0]["content"] == USER_B["content"]

    async def test_rollback_to_is_a_no_op_when_nothing_was_added(self, session):
        await session.add_items([USER_A])
        await rollback_to(session, 1)
        assert len(await session.get_items()) == 1


class TestOverlappingTurnsAreSerialised:
    async def test_second_turn_waits_for_the_first(self, session):
        """Chainlit spawns a task per message. Without the lock the two runs
        interleave as user A, user B, assistant B, assistant A."""
        order: list[str] = []

        async def turn(label: str, delay: float):
            async with atomic_turn(session):
                order.append(f"{label}-start")
                await session.add_items([{"role": "user", "content": label}])
                await asyncio.sleep(delay)
                await session.add_items([{"role": "assistant", "content": label}])
                order.append(f"{label}-end")

        await asyncio.gather(turn("A", 0.05), turn("B", 0.0))

        # Whichever ran first must finish before the other starts.
        assert order in (
            ["A-start", "A-end", "B-start", "B-end"],
            ["B-start", "B-end", "A-start", "A-end"],
        ), order

        contents = [i["content"] for i in await session.get_items()]
        assert contents in (["A", "A", "B", "B"], ["B", "B", "A", "A"]), contents

    async def test_a_failing_turn_releases_the_lock(self, session):
        with pytest.raises(RuntimeError):
            async with atomic_turn(session):
                raise RuntimeError("boom")

        # Must not deadlock.
        async with atomic_turn(session):
            await session.add_items([USER_A])
        assert len(await session.get_items()) == 1

    async def test_each_session_gets_its_own_lock(self):
        """Two browser sessions must never block each other."""
        one, two = SQLiteSession("one"), SQLiteSession("two")
        assert lock_for(one) is not lock_for(two)
        assert lock_for(one) is lock_for(one)

    async def test_separate_sessions_run_concurrently(self):
        one, two = SQLiteSession("s1"), SQLiteSession("s2")
        running = []

        async def turn(session, label):
            async with atomic_turn(session):
                running.append(label)
                await asyncio.sleep(0.05)

        await asyncio.gather(turn(one, "a"), turn(two, "b"))
        # Both entered; neither waited on the other.
        assert sorted(running) == ["a", "b"]
