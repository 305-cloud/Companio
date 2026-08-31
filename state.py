"""
state.py - the two halves of every turn, per the original loop diagram:

    INPUT --> [ Internal State + External State ] --> BRAIN --> ACTION --> NEW STATE
                       ^                                                       |
                       +-------------------- feeds back in ---------------------+

External State = the Actor's turn: what's happening right now (this
                  turn's raw input/context) -- "working memory."
Internal State = the Observer's accumulated belief: what the companion
                  has learned about this specific human, distilled and
                  editable, not a raw transcript.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from belief import BetaBelief


@dataclass
class ExternalState:
    """The current turn's raw input and context (the Actor's state)."""

    user_id: str
    text: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    image_bytes: Optional[bytes] = None   # optional multimodal attachment for this turn
    image_mime: Optional[str] = None       # e.g. "image/png" -- required if image_bytes is set


@dataclass
class SemanticFact:
    """One distilled, structured, editable fact in the Internal State."""

    id: str
    key: str                   # e.g. "response_style", "topic:onboarding"
    label: str                  # human-readable description
    value: Any                   # underlying structured value
    confidence: float             # 0..1 -- posterior mean once `belief` is set
    updated_at: float = field(default_factory=time.time)
    source_event_ids: List[str] = field(default_factory=list)
    status: str = "active"       # "active" | "pending_confirmation"
    pii: bool = False
    belief: Optional[BetaBelief] = None   # evidence-weighted confidence (see belief.py)
    pending_value: Any = None              # staged value while status == pending_confirmation
    pending_label: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "confidence": round(self.confidence, 3),
            "updated_at": self.updated_at,
            "status": self.status,
            "pii": self.pii,
        }
        if self.belief is not None:
            d["concentration"] = round(self.belief.concentration, 2)
        if self.status == "pending_confirmation":
            d["pending_value"] = self.pending_value
            d["pending_label"] = self.pending_label
        return d


@dataclass
class InternalState:
    """
    The accumulated, distilled belief about one specific user -- the
    thing that makes the companion personal rather than generic.
    """

    user_id: str
    facts: Dict[str, SemanticFact] = field(default_factory=dict)
    consolidation_cycles: int = 0

    def active_facts(self) -> List[SemanticFact]:
        return [f for f in self.facts.values() if f.status == "active"]

    def pending_facts(self) -> List[SemanticFact]:
        return [f for f in self.facts.values() if f.status == "pending_confirmation"]

    def get(self, key: str) -> Optional[SemanticFact]:
        return self.facts.get(key)

    def upsert(self, fact: SemanticFact) -> None:
        self.facts[fact.key] = fact

    def remove(self, key: str) -> bool:
        return self.facts.pop(key, None) is not None

    def as_profile(self) -> List[Dict[str, Any]]:
        return [f.as_dict() for f in self.facts.values()]
