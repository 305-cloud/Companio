"""
stub.py - a deterministic, dependency-free backend for local development
and CI. This is what runs out of the box with no API key.
"""

from __future__ import annotations

from typing import Any, Dict

from llm.base import LLMBackend


class StubBackend(LLMBackend):
    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        facts = context.get("semantic_facts", [])
        domain_knowledge = context.get("domain_knowledge", [])
        user_text = context.get("user_text", "")

        if facts:
            top = sorted(facts, key=lambda f: f["confidence"], reverse=True)[0]
            personalization = f" Given what I know -- {top['label']} -- "
        else:
            personalization = " "

        grounding = f"Referencing: {domain_knowledge[0]}. " if domain_knowledge else ""
        return (
            f"{grounding}{personalization.strip()}here's my take on \"{user_text.strip()}\": "
            f"this looks consistent with your recent pattern; let me know if you'd like more detail."
        )
