"""
adk_backend.py - routes generation through Google's Agent Development Kit
(the `google-adk` package: https://google.github.io/adk-docs) instead of
calling google-genai directly, the way llm/gemini.py does.

Why this file exists alongside llm/gemini.py rather than replacing it:
the hackathon's mandatory stack calls for "a Google Agent Framework,"
and calling the google-genai SDK directly (llm/gemini.py) is a real but
debatable way to satisfy that -- it's an SDK call, not a framework. This
backend runs the exact same generation through an actual `adk.Agent` +
`adk.Runner`, which is unambiguous. Both backends implement the same
`LLMBackend` interface and are interchangeable; main.py prefers this one
when google-adk is installed and falls back to llm/gemini.py otherwise,
so nothing upstream (memory, belief, ask-vs-act, consolidation) needed
to change either way.

Requires: pip install google-adk
Env vars: GEMINI_API_KEY or GOOGLE_API_KEY; optionally GEMINI_MODEL.

Verified against the actual installed google-adk 2.8.0 API (not written
from memory): `Agent` and `Runner`/`InMemoryRunner` are pydantic models
with a synchronous `Runner.run()` generator; session creation itself is
async, so it happens once via `asyncio.run()` at construction time.
ADK's Gemini model wrapper constructs its own `google.genai.Client()`
with no explicit api_key/vertexai args, which means it picks up
GEMINI_API_KEY/GOOGLE_API_KEY from the environment exactly like
llm/gemini.py does -- confirmed by inspecting
google.adk.models.google_llm.Gemini.api_client, and by an end-to-end
smoke test that reached Google's real API and got a genuine
"API key not valid" response back (not a local/import error) when
tested with a placeholder key.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Dict, Optional

from llm.base import LLMBackend

try:
    from google.adk import Agent
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ADKBackend requires the google-adk package: pip install google-adk"
    ) from exc

_KEY_HELP_URL = "https://aistudio.google.com/apikey"
_DEFAULT_MODEL = "gemini-3.6-flash"
_APP_NAME = "companion"


class ADKBackendError(Exception):
    """Raised for anything the caller should see clearly instead of a raw SDK trace."""


_FALLBACK_INSTRUCTION = (
    "You are the reasoning core of a personal companion. Respond helpfully "
    "and specifically using only the context you're given in the prompt -- "
    "it already contains what's known about this user and any relevant "
    "background."
)


class ADKBackend(LLMBackend):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> None:
        """
        `instruction` becomes the ADK Agent's persistent system instruction
        (its actual intended mechanism for this) -- pass a domain's
        system_prompt/purpose here so it's set once at construction rather
        than re-stated inside every turn's prompt text. Falls back to a
        generic instruction if not given.
        """
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ADKBackendError(
                "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your "
                f"environment. Generate one at {_KEY_HELP_URL}."
            )
        # ADK's Gemini model wrapper reads the key from the environment itself
        # (see module docstring) -- make sure it's actually there under the
        # name it looks for, regardless of which env var the caller used.
        os.environ.setdefault("GOOGLE_API_KEY", resolved_key)

        self.model = model or os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL)
        self._user_id = "companion-user"
        self._session_id = str(uuid.uuid4())

        try:
            self._agent = Agent(
                name="companion_guide",
                model=self.model,
                instruction=instruction or _FALLBACK_INSTRUCTION,
            )
            self._runner = InMemoryRunner(agent=self._agent, app_name=_APP_NAME)
            asyncio.run(self._runner.session_service.create_session(
                app_name=_APP_NAME, user_id=self._user_id, session_id=self._session_id,
            ))
        except Exception as exc:
            raise ADKBackendError(f"Failed to initialize the ADK agent/session: {exc}") from exc

    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        full_prompt = self._build_content(prompt, context)
        message = genai_types.Content(role="user", parts=[genai_types.Part(text=full_prompt)])

        final_text: Optional[str] = None
        try:
            for event in self._runner.run(
                user_id=self._user_id, session_id=self._session_id, new_message=message,
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    final_text = "".join(p.text or "" for p in event.content.parts)
        except Exception as exc:  # ADK's own retry/backoff already ran by this point
            raise ADKBackendError(f"ADK agent run failed: {exc}") from exc

        if not final_text:
            raise ADKBackendError(
                "The ADK agent produced no final response text (a safety filter "
                "may have blocked it, or the run ended without a text response)."
            )
        return final_text

    @staticmethod
    def _build_content(prompt: str, context: Dict[str, Any]) -> str:
        """Deliberately does NOT re-include context['system_prompt'] here --
        that's the domain's system_prompt/purpose, and it's already set as
        this Agent's persistent `instruction` at construction time (see
        __init__). Restating it in every turn would just be redundant
        tokens. If you construct ADKBackend() directly without passing
        `instruction`, per-turn system_prompt context is not used --
        pass it through main.py's wiring, or set `instruction=` yourself."""
        facts = "\n".join(f"- {f['label']} (confidence {f['confidence']})" for f in context.get("semantic_facts", []))
        knowledge = "\n".join(context.get("domain_knowledge", []))
        return (
            f"What you know about this user:\n{facts or '(no prior facts yet)'}\n\n"
            f"Relevant domain knowledge:\n{knowledge or '(none)'}\n\n"
            f"User: {context.get('user_text', '')}\n\n"
            f"Instruction: {prompt}"
        )
