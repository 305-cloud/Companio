"""
Both llm/gemini.py and llm/adk_backend.py need optional packages
(google-genai / google-adk) and a real network+key to fully exercise, so
these tests only cover what's safe and meaningful without either: the
fail-fast missing-key path. Skipped entirely if the optional package
isn't installed, rather than failing the suite for people who haven't
`pip install`ed them.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest  # noqa: E402


def _clear_key_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_gemini_backend_fails_fast_on_missing_key(monkeypatch):
    pytest.importorskip("google.genai")
    from llm.gemini import GeminiBackend, GeminiBackendError

    _clear_key_env(monkeypatch)
    with pytest.raises(GeminiBackendError, match="No Gemini API key found"):
        GeminiBackend()


def test_adk_backend_fails_fast_on_missing_key(monkeypatch):
    pytest.importorskip("google.adk")
    from llm.adk_backend import ADKBackend, ADKBackendError

    _clear_key_env(monkeypatch)
    with pytest.raises(ADKBackendError, match="No Gemini API key found"):
        ADKBackend()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
