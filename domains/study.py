"""
study.py - a second, unrelated domain: proves the framework is genuinely
domain-agnostic. Its domain_knowledge is a mix of curated study-science
facts and real, messy data ingested from an actual student group chat --
see ingest_whatsapp.py for the pipeline (parse -> chunk -> anonymize ->
safety-filter) that produced study_domain_knowledge.json.
"""

from __future__ import annotations

from domain import DomainConfig
from domains.domain_knowledge_loader import load_domain_knowledge

_CURATED_KNOWLEDGE = [
    "Spaced repetition improves long-term retention more than re-reading.",
    "Concepts explained with a concrete example are retained better than abstract definitions alone.",
]

# real, messy WhatsApp study-group chat, ingested and anonymized by
# ingest_whatsapp.py -- see that file's docstring for the full pipeline
_INGESTED_KNOWLEDGE = load_domain_knowledge("study_domain_knowledge.json")

STUDY_DOMAIN = DomainConfig(
    name="study_companion",
    purpose="Help this specific user understand a dense document, adapting to what they already grasp.",
    required_slots=["topic", "difficulty"],
    clarifying_question_bank=[
        "Is this your first time seeing this concept, or a refresher?",
        "Would you like a quick summary first, or the full detail right away?",
    ],
    system_prompt=(
        "You are a study companion. Adapt explanations to what this user has "
        "already struggled with or mastered, rather than repeating a fixed script. "
        "Some of your domain knowledge comes from a real (anonymized) student "
        "group chat -- casual, mixed-language chatter about scheduling, course "
        "numbers, and group logistics, not formal course material."
    ),
    domain_knowledge=_CURATED_KNOWLEDGE + _INGESTED_KNOWLEDGE,
)
