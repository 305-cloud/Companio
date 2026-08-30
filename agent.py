"""
agent.py - the Agent. The Brain. The whole thing wired together.

    INPUT --> [ Internal State + External State ] --> BRAIN --> ACTION --> NEW STATE
                       ^                                                       |
                       +-------------------- feeds back in ---------------------+

Five stages, each delegated to a sub-agent:
    Elicit        -> Clarifier   (ask vs. act, confidence-scored, self-improving)
    Retrieve      -> Retriever   (semantic + episodic + domain knowledge)
    Act / Guide   -> Guide       (LLM-backed, personalized response)
    Capture       -> FeedbackListener (explicit + implicit signals)
    Consolidate   -> Consolidator (episodic -> semantic Bayesian belief, contradiction handling)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agents.clarifier import Action, Clarifier
from agents.feedback import FeedbackEvent, FeedbackListener
from agents.guide import Guide
from agents.retriever import Retriever
from domain import DomainConfig
from llm.base import LLMBackend
from llm.stub import StubBackend
from memory.consolidator import Consolidator
from memory.store import (
    SOURCE_AGENT,
    SOURCE_HUMAN,
    SOURCE_TRANSFORMER,
    EpisodicEvent,
    UnifiedMemoryStore,
)
from metrics import AdaptationMetrics
from state import ExternalState, InternalState


@dataclass
class TurnResult:
    response: str
    asked_clarifying: bool
    used_profile: bool
    confidence: float
    pending_confirmations: List[Dict[str, Any]] = field(default_factory=list)
    last_response_event_id: Optional[str] = None


class Companion:
    """One instantiation of the Companion model for one domain."""

    def __init__(
        self,
        domain: DomainConfig,
        llm: Optional[LLMBackend] = None,
        db_path: str = ":memory:",
    ) -> None:
        self.domain = domain
        self.llm = llm or StubBackend()
        self.store = UnifiedMemoryStore(db_path)
        self.clarifier = Clarifier()
        self.retriever = Retriever(self.store)
        self.guide = Guide(self.llm)
        self.feedback_listener = FeedbackListener()
        self.consolidator = Consolidator(self.store)
        self.metrics = AdaptationMetrics()
        self._internal_states: Dict[str, InternalState] = {}
        self._last_decision: Dict[str, Tuple[List[float], bool]] = {}

    # ---------- state management ----------

    def _load_internal_state(self, user_id: str) -> InternalState:
        if user_id not in self._internal_states:
            state = InternalState(user_id=user_id)
            state.facts = self.store.read_semantic(user_id)
            self._internal_states[user_id] = state
        return self._internal_states[user_id]

    # ---------- the loop ----------

    def turn(self, user_id: str, text: str, context: Optional[Dict[str, Any]] = None) -> TurnResult:
        external = ExternalState(user_id=user_id, text=text, context=context or {})
        internal = self._load_internal_state(user_id)

        input_event = EpisodicEvent.new(user_id, SOURCE_HUMAN, "input", {"text": text})
        self.store.write_episodic(input_event)

        retrieval = self.retriever.retrieve(external, internal, self.domain)

        decision = self.clarifier.decide(
            external, self.domain, retrieval.retrieval_score, retrieval.profile_match_strength
        )
        self._last_decision[user_id] = (decision.features, decision.action != Action.ASK)

        pending = [f.as_dict() for f in internal.pending_facts()]

        if decision.action == Action.ASK:
            response_text = decision.question or "Could you tell me more?"
            self.metrics.record_turn(user_id, asked_clarifying=True, used_profile=False)
            resp_event = EpisodicEvent.new(user_id, SOURCE_AGENT, "clarify", {"text": response_text})
            self.store.write_episodic(resp_event)
            return TurnResult(response_text, True, False, decision.confidence, pending, resp_event.id)

        assumption_note = None
        if decision.action == Action.PROCEED_WITH_FLAG:
            assumption_note = "I'm inferring this from limited context so far -- correct me if I'm off."

        guide_response = self.guide.respond(text, retrieval, self.domain, assumption_note)

        transformer_event = EpisodicEvent.new(
            user_id, SOURCE_TRANSFORMER, "generation",
            {"prompt_tokens_context": len(retrieval.domain_knowledge)},
        )
        self.store.write_episodic(transformer_event)

        action_event = EpisodicEvent.new(
            user_id, SOURCE_AGENT, "advice", {"text": guide_response.text},
        )
        self.store.write_episodic(action_event)

        self.metrics.record_turn(user_id, asked_clarifying=False, used_profile=guide_response.used_profile)

        return TurnResult(
            response=guide_response.text,
            asked_clarifying=False,
            used_profile=guide_response.used_profile,
            confidence=decision.confidence,
            pending_confirmations=pending,
            last_response_event_id=action_event.id,
        )

    # ---------- feedback + consolidation ----------

    def give_feedback(
        self,
        user_id: str,
        rating: str,
        fact_update: Optional[Dict[str, Any]] = None,
    ) -> FeedbackEvent:
        event = self.feedback_listener.capture_explicit(rating, fact_update)
        if rating in ("down", "correction"):
            self.metrics.record_correction(user_id)

        # only "up"/"down" judge the ask-vs-act decision itself; a bare
        # "correction" is teaching the memory system a fact, not saying
        # the previous turn was wrong to ask (or not ask) -- it shouldn't
        # push a reward into the Clarifier's weights.
        if rating in ("up", "down") and user_id in self._last_decision:
            features, action_was_proceed = self._last_decision[user_id]
            reward = 1.0 if rating == "up" else -1.0
            self.clarifier.learn(features, action_was_proceed, reward)

        if fact_update:
            ep = EpisodicEvent.new(user_id, SOURCE_HUMAN, "feedback", fact_update)
            self.store.write_episodic(ep)
        return event

    def consolidate(self, user_id: str) -> InternalState:
        internal = self._load_internal_state(user_id)
        new_events = self.store.read_episodic(user_id, limit=50, unconsumed_only=True)
        structured = [e for e in new_events if "key" in e.payload]
        self.consolidator.consolidate(user_id, structured, internal)
        return internal

    def resolve_pending(self, user_id: str, key: str, accept_update: bool) -> None:
        internal = self._load_internal_state(user_id)
        self.consolidator.resolve_pending(internal, key, accept_update)
        fact = internal.get(key)
        if fact:
            self.store.upsert_semantic(user_id, fact)

    # ---------- privacy / transparency surface ----------

    def profile(self, user_id: str) -> List[Dict[str, Any]]:
        return self._load_internal_state(user_id).as_profile()

    def forget(self, user_id: str, key: str) -> bool:
        internal = self._load_internal_state(user_id)
        internal.remove(key)
        return self.store.delete_semantic(user_id, key)

    def live_feed(self, user_id: str, n: int = 8) -> List[Dict[str, Any]]:
        return [
            {"source": e.source, "event_type": e.event_type, "bits": e.bits, "ts": e.ts}
            for e in self.store.live_feed_tail(user_id, n)
        ]

    def adaptation_metrics(self, user_id: str) -> List[Dict[str, Any]]:
        return self.metrics.session_trend(user_id)

    def new_session(self, user_id: str) -> None:
        self.metrics.new_session(user_id, session_id=str(uuid.uuid4())[:8])
