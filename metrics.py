"""
metrics.py - turns "it adapts" from a claim into a number.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SessionStats:
    session_id: str
    started_at: float = field(default_factory=time.time)
    turns: int = 0
    clarifying_questions: int = 0
    profile_influenced_responses: int = 0
    corrections: int = 0


class AdaptationMetrics:
    """In-memory metrics tracker, keyed by user_id -> list of sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, List[SessionStats]] = defaultdict(list)

    def new_session(self, user_id: str, session_id: str) -> SessionStats:
        stats = SessionStats(session_id=session_id)
        self._sessions[user_id].append(stats)
        return stats

    def current(self, user_id: str) -> SessionStats:
        sessions = self._sessions[user_id]
        if not sessions:
            return self.new_session(user_id, f"session-{len(sessions) + 1}")
        return sessions[-1]

    def record_turn(self, user_id: str, asked_clarifying: bool, used_profile: bool) -> None:
        s = self.current(user_id)
        s.turns += 1
        if asked_clarifying:
            s.clarifying_questions += 1
        if used_profile:
            s.profile_influenced_responses += 1

    def record_correction(self, user_id: str) -> None:
        self.current(user_id).corrections += 1

    def profile_influence_rate(self, user_id: str) -> float:
        s = self.current(user_id)
        return round(s.profile_influenced_responses / s.turns, 3) if s.turns else 0.0

    def session_trend(self, user_id: str) -> List[Dict]:
        out = []
        for s in self._sessions[user_id]:
            out.append({
                "session_id": s.session_id,
                "turns": s.turns,
                "clarifying_questions": s.clarifying_questions,
                "clarifying_rate": round(s.clarifying_questions / s.turns, 3) if s.turns else 0.0,
                "profile_influence_rate": round(s.profile_influenced_responses / s.turns, 3) if s.turns else 0.0,
                "corrections": s.corrections,
            })
        return out
