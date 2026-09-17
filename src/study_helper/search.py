"""A web search tool for grounding specific facts.

The SDK ships a built-in `WebSearchTool`, but it is a *hosted* tool: it runs
inside OpenAI's Responses API. We talk Chat Completions to a third-party
endpoint, so the SDK rejects it outright ("Hosted tools are not supported with
the ChatCompletions API"). Hence this custom tool.

Two backends, because the good one is fragile. `ddgs` scrapes DuckDuckGo and
returns current results, but it can rate-limit or break when the markup
changes. Wikipedia's API is stable and keyless but lags on recent events. Try
the first, fall back to the second, and if both fail say so plainly rather
than raising - a tutor that cannot search should keep teaching.
"""

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from agents import function_tool

_USER_AGENT = "study-helper/0.1 (educational project)"
_TIMEOUT_SECONDS = 15

# Returned to the model when every backend fails. It is a normal string, not
# an exception: a failed calculation deserves a retry, a failed search does not.
UNAVAILABLE_PREFIX = "SEARCH_UNAVAILABLE"


@dataclass(slots=True)
class SearchResult:
    title: str
    snippet: str
    url: str


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def _search_ddgs(query: str, max_results: int) -> list[SearchResult]:
    """Current web results. Imported lazily so a broken ddgs cannot break import."""
    from ddgs import DDGS

    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=max_results))

    return [
        SearchResult(
            title=hit.get("title", "").strip(),
            snippet=hit.get("body", "").strip(),
            url=hit.get("href", "").strip(),
        )
        for hit in hits
    ]


def _search_wikipedia(query: str, max_results: int) -> list[SearchResult]:
    """Stable keyless fallback. Weaker on recent events, but it does not scrape."""
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json",
        }
    )
    request = urllib.request.Request(
        f"https://en.wikipedia.org/w/api.php?{params}",
        headers={"User-Agent": _USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        payload = json.load(response)

    return [
        SearchResult(
            title=hit["title"],
            snippet=_strip_html(hit.get("snippet", "")),
            url="https://en.wikipedia.org/wiki/" + urllib.parse.quote(hit["title"].replace(" ", "_")),
        )
        for hit in payload.get("query", {}).get("search", [])
    ]


def _format(results: list[SearchResult]) -> str:
    return "\n".join(
        f"[{number}] {result.title}\n    {result.snippet}\n    {result.url}"
        for number, result in enumerate(results, start=1)
    )


def run_search(query: str, max_results: int = 3) -> str:
    """Search the web and return formatted results. Never raises.

    Returns a string starting with UNAVAILABLE_PREFIX when nothing worked, so
    the caller can keep going instead of handling an exception.
    """
    if not query.strip():
        return f"{UNAVAILABLE_PREFIX}: empty query. Answer from your own knowledge."

    # Resolved from module globals on each call so tests can monkeypatch them.
    backends = (("ddgs", _search_ddgs), ("wikipedia", _search_wikipedia))

    failures: list[str] = []
    for label, backend in backends:
        try:
            results = backend(query, max_results)
        except Exception as err:  # network, rate limit, markup change, bad JSON
            failures.append(f"{label}: {type(err).__name__}")
            continue
        if results:
            return _format(results)
        failures.append(f"{label}: no results")

    return (
        f"{UNAVAILABLE_PREFIX}: {'; '.join(failures)}. "
        "Answer from your own knowledge and say the figure may be out of date."
    )


@function_tool
def web_search(query: str, max_results: int = 3) -> str:
    """Look up current facts on the web to check a specific claim.

    Use this to verify a named fact, figure, date, or measurement - not to
    explain a concept. If it returns SEARCH_UNAVAILABLE, answer from your own
    knowledge instead.

    Args:
        query: What to look up, as plain search keywords.
        max_results: How many results to return, 1 to 5.
    """
    return run_search(query, max_results=max(1, min(max_results, 5)))
