"""
general.py - the default domain: no fixed vertical at all.
"""

from __future__ import annotations

from domain import DomainConfig

GENERAL_DOMAIN = DomainConfig(
    name="general_companion",
    purpose=(
        "Be a general-purpose personal companion: understand this specific "
        "person over time, across whatever they bring, and let what has "
        "actually been learned about them -- not a fixed script -- shape "
        "how you respond."
    ),
    required_slots=[],
    clarifying_question_bank=[
        "Could you tell me a bit more about what you're looking for?",
        "What's the context here -- what are you actually trying to do?",
    ],
    system_prompt=(
        "You are a general-purpose companion AI with no fixed domain. "
        "Adapt to whatever the person brings. Ground your response in what "
        "you've actually learned about them so far, and say plainly when "
        "you're inferring rather than certain."
    ),
    domain_knowledge=[],
)
