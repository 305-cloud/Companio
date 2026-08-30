"""
retriever.py - pulls both halves of context the Brain needs:
  1. semantic + episodic memory (this specific human)
  2. domain knowledge (the "Transformer" / general-knowledge layer, RAG)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from domain import DomainConfig
from memory.store import EpisodicEvent, UnifiedMemoryStore
from state import ExternalState, InternalState, SemanticFact


@dataclass
class RetrievalResult:
    semantic_facts: List[SemanticFact] = field(default_factory=list)
    episodic_events: List[EpisodicEvent] = field(default_factory=list)
    domain_knowledge: List[str] = field(default_factory=list)
    retrieval_score: float = 0.0
    profile_match_strength: float = 0.0


class Retriever:
    def __init__(self, store: UnifiedMemoryStore) -> None:
        self.store = store

    def _search_domain_knowledge(self, query: str, domain: DomainConfig, k: int = 2) -> List[str]:
        query_words = set(query.lower().split())
        scored = []
        for snippet in domain.domain_knowledge:
            overlap = len(query_words & set(snippet.lower().split()))
            if overlap:
                scored.append((overlap, snippet))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:k]]

    def retrieve(self, external: ExternalState, internal: InternalState, domain: DomainConfig) -> RetrievalResult:
        facts = internal.active_facts()
        events = self.store.read_episodic(external.user_id, limit=10)
        knowledge = self._search_domain_knowledge(external.text, domain)

        retrieval_score = min(1.0, len(knowledge) / 2) if domain.domain_knowledge else 0.5
        profile_match_strength = min(1.0, len(facts) / 4)

        return RetrievalResult(
            semantic_facts=facts,
            episodic_events=events,
            domain_knowledge=knowledge,
            retrieval_score=retrieval_score,
            profile_match_strength=profile_match_strength,
        )
