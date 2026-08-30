"""
base.py - the interface every reasoning backend implements.

The Brain never talks to a model directly; it talks to an LLMBackend.
Swapping StubBackend for GeminiBackend (or any other model) touches
exactly one line of wiring in agent.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        """Return a text completion given a prompt and structured context."""
        raise NotImplementedError
