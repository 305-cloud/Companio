"""
guide.py - the Brain's decision/action stage: produces the actual
response, grounded in both retrieved domain knowledge and the user's
personal profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from domain import DomainConfig
from llm.base import LLMBackend
from agents.retriever import RetrievalResult


@dataclass
class GuideResponse:
    text: str
    used_profile: bool


class Guide:
    def __init__(self, llm: LLMBackend) -> None:
        self.llm = llm

    def respond(
        self,
        user_text: str,
        retrieval: RetrievalResult,
        domain: DomainConfig,
        assumption_note: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_mime: Optional[str] = None,
    ) -> GuideResponse:
        context = {
            "system_prompt": domain.system_prompt or domain.purpose,
            "user_text": user_text,
            "semantic_facts": [f.as_dict() for f in retrieval.semantic_facts],
            "domain_knowledge": retrieval.domain_knowledge,
            "image_bytes": image_bytes,
            "image_mime": image_mime,
        }
        text = self.llm.generate(prompt="Respond helpfully and specifically to this user.", context=context)
        if assumption_note:
            text = f"{text}\n\n({assumption_note})"
        return GuideResponse(text=text, used_profile=bool(retrieval.semantic_facts))
