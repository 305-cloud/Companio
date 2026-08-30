"""
clarifier.py - the ask-vs-act decision, made explicit, scored, and
self-improving. See ARCHITECTURE notes in README.md for the full
reasoning; short version:

    p = sigmoid(bias + w . [slot_fill_rate, retrieval_score, profile_match_strength])
    entropy = binary_entropy(p)

    p < tau_lo  OR  entropy > tau_entropy   -> ASK
    p < tau_hi                               -> PROCEED_WITH_FLAG
    otherwise                                 -> PROCEED_SILENT

A per-user AdaptiveGate can also force ASK when a turn deviates sharply
from this person's usual pattern. `learn()` takes a REINFORCE-style
policy-gradient step from real feedback (see agent.py::give_feedback).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from domain import DomainConfig
from state import ExternalState
from gate import AdaptiveGate


class Action(str, Enum):
    ASK = "ask"
    PROCEED_WITH_FLAG = "proceed_with_flag"
    PROCEED_SILENT = "proceed_silent"


@dataclass
class ClarifierResult:
    action: Action
    confidence: float          # p: the model's estimate that acting is appropriate
    entropy: float = 0.0
    question: Optional[str] = None
    features: List[float] = field(default_factory=list)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _binary_entropy(p: float) -> float:
    eps = 1e-9
    p = min(max(p, eps), 1 - eps)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


class Clarifier:
    def __init__(
        self,
        weights: Optional[List[float]] = None,
        bias: float = -1.6,
        tau_lo: float = 0.35,
        tau_hi: float = 0.75,
        tau_entropy: float = 0.99,
        learning_rate: float = 0.08,
        gate_alpha: float = 0.1,
        gate_delta: float = 0.25,
    ) -> None:
        self.weights = list(weights) if weights is not None else [1.2, 1.0, 1.0]
        self.bias = bias
        self.tau_lo = tau_lo
        self.tau_hi = tau_hi
        self.tau_entropy = tau_entropy
        self.learning_rate = learning_rate
        self._gate_alpha = gate_alpha
        self._gate_delta = gate_delta
        self._gates: Dict[str, AdaptiveGate] = {}

    def _gate_for(self, user_id: str) -> AdaptiveGate:
        if user_id not in self._gates:
            self._gates[user_id] = AdaptiveGate(alpha=self._gate_alpha, delta=self._gate_delta)
        return self._gates[user_id]

    def _slot_fill_rate(self, external: ExternalState, domain: DomainConfig) -> float:
        if not domain.required_slots:
            return 1.0
        text = external.text.lower()
        filled = sum(1 for slot in domain.required_slots if slot.lower() in text or slot in external.context)
        return filled / len(domain.required_slots)

    def _score(self, features: List[float]) -> float:
        z = self.bias + sum(w * x for w, x in zip(self.weights, features))
        return _sigmoid(z)

    def decide(
        self,
        external: ExternalState,
        domain: DomainConfig,
        retrieval_score: float,
        profile_match_strength: float,
    ) -> ClarifierResult:
        slot_fill = self._slot_fill_rate(external, domain)
        features = [slot_fill, retrieval_score, profile_match_strength]

        p = self._score(features)
        entropy = _binary_entropy(p)
        gate_fires, _deviation = self._gate_for(external.user_id).step(p)

        question = domain.clarifying_question_bank[0] if domain.clarifying_question_bank else (
            "Could you tell me a bit more about that before I respond?"
        )

        if p < self.tau_lo or entropy > self.tau_entropy or (gate_fires and p < self.tau_hi):
            return ClarifierResult(Action.ASK, p, entropy, question, features)
        if p < self.tau_hi:
            return ClarifierResult(Action.PROCEED_WITH_FLAG, p, entropy, None, features)
        return ClarifierResult(Action.PROCEED_SILENT, p, entropy, None, features)

    def learn(self, features: List[float], action_was_proceed: bool, reward: float) -> None:
        """
        One REINFORCE-style policy-gradient step. reward in [-1, 1]:
        +1 = the decision (ask or act) turned out right, -1 = it didn't.
        """
        p = self._score(features)
        a = 1.0 if action_was_proceed else 0.0
        grad_scale = reward * (a - p)
        self.weights = [w + self.learning_rate * grad_scale * x for w, x in zip(self.weights, features)]
        self.bias += self.learning_rate * grad_scale

    def as_dict(self) -> dict:
        return {"weights": [round(w, 4) for w in self.weights], "bias": round(self.bias, 4)}
