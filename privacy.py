"""
privacy.py - the trust layer.

Memory is data about a real person. Before anything is written to the
unified database, it passes through a lightweight PII check.
"""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"\b(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

_PATTERNS = [_EMAIL_RE, _PHONE_RE, _CARD_RE, _SSN_RE]


def contains_pii(text: str) -> bool:
    return any(p.search(text) for p in _PATTERNS)


def redact(text: str) -> str:
    redacted = text
    for pattern in _PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted
