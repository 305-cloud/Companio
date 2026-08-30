"""
consolidator.py - the Consolidator sub-agent: episodic evidence,
distilled into semantic belief, with contradiction treated as measured
surprise rather than silently overwritten or gated by a fixed
`confidence >= 0.6` cutoff.

Surprise is measured as KL(updated_belief || existing_belief) -- how
much would folding the new evidence in actually move the belief? That
number is naturally scaled by how much evidence the belief already has,
so one constant SURPRISE_THRESHOLD works for a fact seen once and a fact
seen fifty times alike.

A contradiction never overwrites silently: the proposed value is staged
on `pending_value`/`pending_label` and the fact's status becomes
"pending_confirmation" until `resolve_pending` is called.
"""

from __future__ import annotations

import uuid
from typing import List

from state import InternalState, SemanticFact
from belief import BetaBelief
from memory.store import EpisodicEvent, UnifiedMemoryStore

SURPRISE_THRESHOLD = 0.15   # nats
DECAY_RHO = 0.97             # per-consolidation-cycle relaxation toward the uninformative prior


class Consolidator:
    def __init__(self, store: UnifiedMemoryStore) -> None:
        self.store = store

    def consolidate(
        self,
        user_id: str,
        new_events: List[EpisodicEvent],
        internal_state: InternalState,
    ) -> InternalState:
        """episodic -> semantic, with contradiction handling and decay."""

        for fact in internal_state.active_facts():
            if fact.belief is not None:
                fact.belief.decay(DECAY_RHO)
                fact.confidence = fact.belief.mean

        for event in new_events:
            key = event.payload.get("key")
            if not key:
                continue
            label = event.payload.get("label", key)
            value = event.payload.get("value")
            weight = float(event.payload.get("confidence", 1.0))  # evidence strength, not a probability

            existing = internal_state.get(key)
            if existing is None:
                belief = BetaBelief()
                belief.update(agree=True, weight=weight)
                internal_state.upsert(SemanticFact(
                    id=str(uuid.uuid4()), key=key, label=label, value=value,
                    confidence=belief.mean, belief=belief, source_event_ids=[event.id],
                    pii=event.pii,
                ))
                continue

            if existing.belief is None:
                existing.belief = BetaBelief(alpha=1 + existing.confidence, beta=1 + (1 - existing.confidence))

            agrees = existing.value == value
            if agrees:
                existing.belief.update(agree=True, weight=weight)
                existing.confidence = existing.belief.mean
                existing.label = label
                existing.source_event_ids.append(event.id)
                existing.status = "active"
                continue

            surprise = existing.belief.shift_surprise(agree=False, weight=weight)
            if surprise > SURPRISE_THRESHOLD:
                existing.status = "pending_confirmation"
                existing.pending_value = value
                existing.pending_label = label
                existing.source_event_ids.append(event.id)
                continue

            existing.value = value
            existing.label = label
            existing.belief.update(agree=False, weight=weight)
            existing.confidence = existing.belief.mean
            existing.source_event_ids.append(event.id)
            existing.status = "active"

        self.store.mark_consumed([e.id for e in new_events])
        for fact in internal_state.facts.values():
            self.store.upsert_semantic(user_id, fact)

        internal_state.consolidation_cycles += 1
        return internal_state

    def resolve_pending(self, internal_state: InternalState, key: str, accept_update: bool) -> None:
        """Called once the user answers the one-time confirmation prompt."""
        fact = internal_state.get(key)
        if not fact or fact.status != "pending_confirmation":
            return
        if fact.belief is None:
            fact.belief = BetaBelief(alpha=1 + fact.confidence, beta=1 + (1 - fact.confidence))

        if accept_update:
            fact.value = fact.pending_value
            fact.label = fact.pending_label or fact.label
            fact.belief.update(agree=True, weight=1.0)
        else:
            fact.belief.update(agree=True, weight=0.5)

        fact.confidence = fact.belief.mean
        fact.pending_value = None
        fact.pending_label = None
        fact.status = "active"
