"""
gemini.py - production backend, hardened so it fails predictably and
recovers gracefully instead of surfacing a raw SDK stack trace
mid-conversation.

Requires: pip install google-genai
Env vars: GEMINI_API_KEY or GOOGLE_API_KEY; optionally GEMINI_MODEL.

IMPORTANT: Gemini model ids get retired over time (e.g. `gemini-2.5-flash`
returning 404 "no longer available to new users"). Confirm the current
valid model id for your account in Google AI Studio and set it via
GEMINI_MODEL rather than trusting any hardcoded default blindly.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import httpx

from llm.base import LLMBackend

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "GeminiBackend requires the google-genai package: pip install google-genai"
    ) from exc

_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BACKOFF_SECONDS = (1, 2, 4)
_KEY_HELP_URL = "https://aistudio.google.com/apikey"
_DEFAULT_MODEL = "gemini-3.6-flash"


class GeminiBackendError(Exception):
    """Raised for anything the caller should see clearly instead of a raw SDK trace."""


class GeminiBackend(LLMBackend):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise GeminiBackendError(
                "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your "
                f"environment. Generate one at {_KEY_HELP_URL}."
            )
        self.model = model or os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL)
        self.client = genai.Client(
            api_key=resolved_key,
            http_options=genai_types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )

    def ping(self) -> bool:
        """Isolates 'does the key/model work at all' from the rest of the
        companion loop, so key/network problems are never confused with
        memory or consolidation bugs."""
        response = self._call("ping", genai_types.GenerateContentConfig(max_output_tokens=8))
        return bool(self._extract_text(response))

    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        system_prompt = context.get("system_prompt", "")
        user_content = self._build_content(prompt, context)
        config = genai_types.GenerateContentConfig(system_instruction=system_prompt) if system_prompt else None
        response = self._call(user_content, config)
        return self._extract_text(response)

    # ---------- internals ----------

    def _call(self, contents: str, config: Optional[Any]):
        last_error: Optional[BaseException] = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self.client.models.generate_content(model=self.model, contents=contents, config=config)
            except genai_errors.APIError as exc:
                last_error = exc
                code = getattr(exc, "code", None)
                if code not in _RETRYABLE_CODES or attempt == _MAX_RETRIES:
                    raise GeminiBackendError(f"Gemini API call failed (code={code}): {exc}") from exc
                time.sleep(_BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)])
            except httpx.TransportError as exc:
                # network/DNS-level failure -- never reached Google's servers at all
                last_error = exc
                if attempt == _MAX_RETRIES:
                    raise GeminiBackendError(
                        "Couldn't reach Gemini's servers -- this looks like a local network/DNS "
                        "problem (check your internet connection, VPN, or firewall), not a code "
                        f"or API-key issue. Underlying error: {exc}"
                    ) from exc
                time.sleep(_BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)])
        raise GeminiBackendError(f"Gemini API call failed after {_MAX_RETRIES} retries: {last_error}")

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return text
        candidates = getattr(response, "candidates", None) or []
        finish_reason = candidates[0].finish_reason if candidates else None
        prompt_feedback = getattr(response, "prompt_feedback", None)
        raise GeminiBackendError(
            "Gemini returned no text (a safety filter likely blocked the response). "
            f"finish_reason={finish_reason} prompt_feedback={prompt_feedback}"
        )

    @staticmethod
    def _build_content(prompt: str, context: Dict[str, Any]) -> str:
        facts = "\n".join(f"- {f['label']} (confidence {f['confidence']})" for f in context.get("semantic_facts", []))
        knowledge = "\n".join(context.get("domain_knowledge", []))
        return (
            f"What you know about this user:\n{facts or '(no prior facts yet)'}\n\n"
            f"Relevant domain knowledge:\n{knowledge or '(none)'}\n\n"
            f"User: {context.get('user_text', '')}\n\n"
            f"Instruction: {prompt}"
        )
