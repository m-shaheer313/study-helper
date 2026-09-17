"""Shared test setup.

Offline tests build agents to inspect their structure - which tools and
guardrails are attached, which model slot they use - but never call a model.
`build_model()` still insists on a key, so without one those tests fail with
SystemExit for a reason that has nothing to do with what they assert.

A placeholder key keeps the offline suite genuinely runnable with no
credentials. A real key from .env is left alone, so the live tests still work.
"""

import os

import pytest

from study_helper import config


@pytest.fixture(autouse=True)
def offline_api_key(monkeypatch):
    # Must yield on every path: an early return in a generator fixture makes
    # pytest raise "did not yield a value" for every test that uses it.
    patched = not os.environ.get("STUDY_HELPER_API_KEY")
    if patched:
        monkeypatch.setenv("STUDY_HELPER_API_KEY", "offline-placeholder")
        config.build_model.cache_clear()
        config.build_structured_model.cache_clear()

    yield

    if patched:
        config.build_model.cache_clear()
        config.build_structured_model.cache_clear()
