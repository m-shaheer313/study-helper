"""Tests for the web search tool.

Backend behaviour is tested offline with monkeypatching - hitting the real
network in unit tests makes them slow and flaky. The `live` tests check
whether the model actually reaches for the tool, which is the part no amount
of unit testing can tell you.
"""

import pytest

from study_helper import search
from study_helper.search import UNAVAILABLE_PREFIX, SearchResult, run_search, web_search

FAKE_HITS = [SearchResult(title="Perseverance", snippet="A NASA rover.", url="https://example.org/p")]


class TestFormatting:
    def test_results_are_numbered_with_title_snippet_and_url(self, monkeypatch):
        monkeypatch.setattr(search, "_search_ddgs", lambda q, n: FAKE_HITS)
        out = run_search("mars rover")
        assert "[1] Perseverance" in out
        assert "A NASA rover." in out
        assert "https://example.org/p" in out

    def test_empty_query_is_refused_without_a_network_call(self, monkeypatch):
        def explode(*_args):
            raise AssertionError("should not have searched")

        monkeypatch.setattr(search, "_search_ddgs", explode)
        monkeypatch.setattr(search, "_search_wikipedia", explode)
        assert run_search("   ").startswith(UNAVAILABLE_PREFIX)


class TestFallback:
    def test_wikipedia_is_used_when_ddgs_raises(self, monkeypatch):
        def broken(*_args):
            raise RuntimeError("rate limited")

        monkeypatch.setattr(search, "_search_ddgs", broken)
        monkeypatch.setattr(search, "_search_wikipedia", lambda q, n: FAKE_HITS)
        assert "Perseverance" in run_search("mars rover")

    def test_wikipedia_is_used_when_ddgs_returns_nothing(self, monkeypatch):
        monkeypatch.setattr(search, "_search_ddgs", lambda q, n: [])
        monkeypatch.setattr(search, "_search_wikipedia", lambda q, n: FAKE_HITS)
        assert "Perseverance" in run_search("mars rover")

    def test_both_failing_returns_a_message_not_an_exception(self, monkeypatch):
        def broken(*_args):
            raise RuntimeError("down")

        monkeypatch.setattr(search, "_search_ddgs", broken)
        monkeypatch.setattr(search, "_search_wikipedia", broken)
        out = run_search("mars rover")
        assert out.startswith(UNAVAILABLE_PREFIX)
        # The model needs to be told what to do instead.
        assert "your own knowledge" in out

    def test_failure_message_names_the_backends_that_failed(self, monkeypatch):
        def broken(*_args):
            raise RuntimeError("down")

        monkeypatch.setattr(search, "_search_ddgs", broken)
        monkeypatch.setattr(search, "_search_wikipedia", lambda q, n: [])
        out = run_search("mars rover")
        assert "ddgs" in out and "wikipedia" in out


class TestToolWiring:
    def test_tool_is_registered_with_a_description(self):
        assert web_search.name == "web_search"
        assert web_search.description

    def test_science_tutor_has_search(self):
        from study_helper.roles.science_tutor import build_science_tutor

        assert "web_search" in [t.name for t in build_science_tutor().tools]

    def test_math_tutor_does_not_have_search(self):
        from study_helper.roles.math_tutor import build_math_tutor

        assert "web_search" not in [t.name for t in build_math_tutor().tools]


@pytest.mark.live
class TestRealBackends:
    def test_wikipedia_backend_returns_results(self):
        results = search._search_wikipedia("Perseverance rover", 3)
        assert results and results[0].title

    def test_ddgs_backend_returns_results(self):
        # Scraped, so this is the test most likely to fail for external reasons.
        results = search._search_ddgs("NASA Mars rover", 3)
        assert results and results[0].title


def _search_calls(result) -> list[str]:
    from agents.items import ToolCallItem

    names = [
        str(getattr(i.raw_item, "name", None) or i.raw_item.get("name", ""))
        for i in result.new_items
        if isinstance(i, ToolCallItem)
    ]
    return [n for n in names if "web_search" in n]


@pytest.mark.live
class TestModelUsesSearchAppropriately:
    async def test_searches_for_a_recent_factual_question(self):
        from agents import Runner

        from study_helper.roles.science_tutor import build_science_tutor

        tutor = build_science_tutor().clone(output_guardrails=[])
        result = await Runner.run(tutor, "What is the latest Mars rover discovery?")
        assert _search_calls(result), "Science Tutor answered a recency question without searching"

    async def test_does_not_search_for_a_conceptual_question(self):
        from agents import Runner

        from study_helper.roles.science_tutor import build_science_tutor

        tutor = build_science_tutor().clone(output_guardrails=[])
        result = await Runner.run(tutor, "Why does ice float on water?")
        assert not _search_calls(result), "searched for a concept it should simply explain"

    async def test_search_survives_the_handoff(self):
        """The calculator silently stopped working through the orchestrator.
        Do not assume the same input_filter fix covers this tool - verify it."""
        from agents import Runner

        from study_helper.roles.orchestrator import build_orchestrator

        result = await Runner.run(
            build_orchestrator(), "What is the latest Mars rover discovery?"
        )
        assert result.last_agent.name == "Science Tutor"
        assert _search_calls(result), "no web_search call after handoff"
