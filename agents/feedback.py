"""
feedback.py - captures both explicit and implicit signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class FeedbackEvent:
    explicit: Optional[str] = None          # "up" | "down" | "correction"
    correction_payload: Optional[Dict[str, Any]] = None
    implicit_signals: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_correction(self) -> bool:
        return self.explicit == "correction"


class FeedbackListener:
    def capture_explicit(self, rating: str, correction_payload: Optional[Dict[str, Any]] = None) -> FeedbackEvent:
        return FeedbackEvent(explicit=rating, correction_payload=correction_payload)

    def infer_implicit(self, repeated_question: bool = False, time_on_step_outlier: bool = False) -> Dict[str, Any]:
        signals = {}
        if repeated_question:
            signals["repeated_question"] = True
        if time_on_step_outlier:
            signals["time_on_step_outlier"] = True
        return signals
