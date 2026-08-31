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


def test_gemini_build_content_stays_a_plain_string_with_no_image():
    """No image attached -- confirms multimodal support didn't change the
    plain-text path any existing caller/test relies on."""
    pytest.importorskip("google.genai")
    from llm.gemini import GeminiBackend

    context = {"semantic_facts": [], "domain_knowledge": [], "user_text": "hey"}
    content = GeminiBackend._build_content("Respond.", context)
    assert isinstance(content, str)


def test_gemini_build_content_attaches_image_part_when_present():
    """Verified against the real installed google-genai API: contents may
    be a list mixing a bare str (auto-wrapped as a text Part) with a
    types.Part -- see llm/gemini.py's _build_content docstring."""
    pytest.importorskip("google.genai")
    from google.genai import types as genai_types
    from llm.gemini import GeminiBackend

    context = {
        "semantic_facts": [], "domain_knowledge": [], "user_text": "what's in this?",
        "image_bytes": b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, "image_mime": "image/png",
    }
    content = GeminiBackend._build_content("Respond.", context)
    assert isinstance(content, list)
    assert isinstance(content[0], str)
    assert isinstance(content[1], genai_types.Part)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
