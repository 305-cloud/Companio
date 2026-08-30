"""
domain.py - makes the companion domain-agnostic.

A DomainConfig is how you point the same Brain/Loop/Memory machinery at
any vertical -- a study companion, a wellness check-in, a UX helper, a
finance tracker -- without touching any other file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class DomainConfig:
    name: str
    purpose: str                     # one sentence: why this agent exists
    required_slots: List[str] = field(default_factory=list)
    clarifying_question_bank: List[str] = field(default_factory=list)
    system_prompt: str = ""          # base instructions for the Guide/LLM
    fact_key_extractor: Optional[Callable[[str], str]] = None
    domain_knowledge: List[str] = field(default_factory=list)
