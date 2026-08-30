"""
domain_knowledge_loader.py - drop-in replacement for a hardcoded
domain_knowledge=[...] list in a DomainConfig: loads it from a JSON file
sitting next to this module instead, so ingestion pipelines like
ingest_whatsapp.py can regenerate it without anyone touching study.py.

Usage in domains/study.py:

    from domains.domain_knowledge_loader import load_domain_knowledge

    STUDY_DOMAIN = DomainConfig(
        ...,
        domain_knowledge=load_domain_knowledge("study_domain_knowledge.json"),
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def load_domain_knowledge(filename: str = "study_domain_knowledge.json") -> List[str]:
    path = Path(__file__).parent / filename
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))
