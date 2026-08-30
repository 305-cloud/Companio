"""
ingest_whatsapp.py - messy-data ingestion for companion's domain_knowledge.

Turns a raw WhatsApp .txt export into clean, chunked, anonymized,
filtered text snippets ready to drop into a DomainConfig.domain_knowledge
list (List[str], matched by keyword overlap in agents/retriever.py).

Pipeline:
  1. Parse export lines into {ts, sender, text} -- merge wrapped multi-line
     messages, separate real messages from system events (joins, group
     creation, encryption notice).
  2. Drop noise: system events, "<Media omitted>", empty/whitespace-only
     text after redaction.
  3. Redact PII using the project's own privacy.py (contains_pii/redact),
     plus a WhatsApp-specific @-mention phone pattern the base regexes
     don't catch (no dashes/spacing: "@923042439149").
  4. Chunk by conversation gap: a new chunk starts whenever the silence
     between two messages exceeds GAP_MINUTES -- the actual topic
     boundary in a group chat, not an arbitrary character count.
  5. Split any chunk that's still too large (word count) so each snippet
     stays dense enough for the retriever's keyword-overlap scoring.
  6. Anonymize every real sender name to a consistent "Student NN" alias
     across the whole corpus (same person -> same alias everywhere,
     including mid-message mentions, not just the "Sender: text" prefix).
  7. Safety-filter: drop any chunk containing profanity/explicit language
     from the demo-facing knowledge base. Nothing is silently discarded --
     filtered chunks are kept in the traceability JSONL with
     flagged_explicit=true, just excluded from domain_knowledge.
  8. Emit two things: a full JSONL (with metadata, for traceability /
     future upgrade to embeddings) and a plain JSON list[str] shaped
     exactly like DomainConfig.domain_knowledge -- only the chunks that
     survived anonymization + the safety filter.

Usage:
    python ingest_whatsapp.py path/to/WhatsApp_Chat_export.txt

Run this from inside companion/domains/ (or anywhere -- it locates
companion/privacy.py relative to its own location either way). Output
files are written next to this script:
    study_chat_chunks.jsonl        (full trace, includes filtered-out chunks)
    study_domain_knowledge.json    (final, anonymized, filtered -- what ships)
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from privacy import contains_pii, redact  # noqa: E402 -- reuse the project's own PII layer

GAP_MINUTES = 45          # silence beyond this = new topic/chunk
MAX_WORDS_PER_CHUNK = 140  # split further if a chunk grows past this
MIN_MESSAGES_PER_CHUNK = 2  # drop chunks with fewer real messages (likely noise)

LINE_RE = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}) - (.*)$"
)
SENDER_RE = re.compile(r"^([^:]{1,40}?): (.*)$")
MENTION_PHONE_RE = re.compile(r"@\d{9,15}\b")
SYSTEM_PATTERNS = re.compile(
    r"(created group|added |removed |left$|changed the group|changed their phone number|"
    r"end-to-end encrypted|changed this group's icon|changed the subject)",
    re.IGNORECASE,
)

# Roman Urdu/Hindi + English profanity and explicit-content markers commonly
# seen in casual group chats. Substring match, case-insensitive -- tuned to
# over-filter rather than under-filter, since this feeds a public demo.
PROFANITY_TERMS = [
    "lund", "land ", " land", "gaand", "gand ", " gand", "chutiya", "chutiye",
    "bhosri", "bhosree", "bhosdi", "bsdk", "behenchod", "bhenchod", "bhenchood",
    "madarchod", "randi", "kutti", "kutta ", "chodu", "chudwao", "chudai",
    "gandu", "harami", "bc ", " bc,", "mc ", " mc,", "fuck", "bitch", " sex ",
]


@dataclass
class Message:
    ts: datetime
    sender: Optional[str]
    text: str


@dataclass
class Chunk:
    chunk_id: str
    start_ts: datetime
    end_ts: datetime
    participants: List[str]
    n_messages: int
    text: str
    flagged_explicit: bool = False


def _parse_ts(date_str: str, time_str: str) -> datetime:
    for fmt in ("%m/%d/%y %H:%M", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(f"{date_str} {time_str}", fmt)
        except ValueError:
            continue
    raise ValueError(f"Unparseable timestamp: {date_str} {time_str}")


def parse_export(path: Path) -> List[Message]:
    messages: List[Message] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip("﻿‎‏ ").rstrip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if m:
            date_str, time_str, rest = m.groups()
            try:
                ts = _parse_ts(date_str, time_str)
            except ValueError:
                if messages:
                    messages[-1].text += " " + line
                continue

            if SYSTEM_PATTERNS.search(rest):
                continue  # group-management noise, not chat content

            sm = SENDER_RE.match(rest)
            if sm:
                sender, text = sm.groups()
                messages.append(Message(ts=ts, sender=sender.strip(), text=text.strip()))
            # else: system message with no "sender: text" shape -> drop
        else:
            # continuation of a wrapped multi-line message
            if messages:
                messages[-1].text += " " + line
    return messages


def clean_text(text: str) -> str:
    text = MENTION_PHONE_RE.sub("[REDACTED]", text)
    text = redact(text)
    text = text.strip()
    if text in {"", "<Media omitted>", "This message was deleted", "null"}:
        return ""
    return text


def to_chunks(messages: List[Message]) -> List[Chunk]:
    groups: List[List[Message]] = []
    current: List[Message] = []
    for msg in messages:
        cleaned = clean_text(msg.text)
        if not cleaned:
            continue
        msg.text = cleaned
        if current and (msg.ts - current[-1].ts) > timedelta(minutes=GAP_MINUTES):
            groups.append(current)
            current = []
        current.append(msg)
    if current:
        groups.append(current)

    chunks: List[Chunk] = []
    for group in groups:
        for sub in _split_by_word_budget(group):
            if len(sub) < MIN_MESSAGES_PER_CHUNK:
                continue
            participants = sorted({m.sender for m in sub if m.sender})
            body = "\n".join(f"{m.sender}: {m.text}" for m in sub)
            chunks.append(Chunk(
                chunk_id=f"chunk_{len(chunks):04d}",
                start_ts=sub[0].ts,
                end_ts=sub[-1].ts,
                participants=participants,
                n_messages=len(sub),
                text=body,
            ))
    return chunks


def _split_by_word_budget(group: List[Message]) -> List[List[Message]]:
    """Budget counts sender-label words too, since that's what actually
    ends up in the rendered chunk text ('Sender: message')."""
    out, current, count = [], [], 0
    for msg in group:
        sender_words = len(msg.sender.split()) if msg.sender else 0
        w = len(msg.text.split()) + sender_words
        if current and count + w > MAX_WORDS_PER_CHUNK:
            out.append(current)
            current, count = [], 0
        current.append(msg)
        count += w
    if current:
        out.append(current)
    return out


# tokens too generic/common to safely treat as an identifying name on
# their own (shared suffix, or a normal Urdu/English word that happens to
# also be someone's first token) -- extend this list if you spot more
# false positives after running the pipeline on your own export.
GENERIC_NAME_TOKENS = {"ku", "g-1", "afzal", "ghazi"}

# Names mentioned in message bodies that never appear as a sender --
# anonymize_names can't discover these on its own (nothing to derive them
# from). List any you spot after a first pass and re-run; group spelling
# variants of the same person into one inner list so they get ONE alias,
# not one each. Matched as whole words, case-insensitive.
EXTRA_NAMES_TO_ANONYMIZE = [
    ["Zainab", "Zainb"],
    ["Aagzia", "aghzia"],
    ["hina"],
    ["asif"],
]


def anonymize_names(chunks: List[Chunk]) -> List[Chunk]:
    """Every real sender name becomes a consistent 'Student NN' alias
    across the whole corpus -- same person, same alias, everywhere. This
    matches both the full sender name ('Ali Ku') AND its individual
    tokens ('Ali') so casual first-name mentions inside a message body
    get caught too, not just the 'Sender: text' prefix -- group chat
    text refers to people by first name far more often than by full name.
    """
    seen_order: List[str] = []
    for c in chunks:
        for p in c.participants:
            if p not in seen_order:
                seen_order.append(p)

    full_alias = {name: f"Student {i + 1:02d}" for i, name in enumerate(seen_order)}

    # build a token -> alias map, skipping tokens that are too generic or
    # ambiguous (shared across multiple different people) to safely replace
    token_owners: dict[str, set] = {}
    for name, alias in full_alias.items():
        for token in name.split():
            if token.lower() in GENERIC_NAME_TOKENS or len(token) < 3:
                continue
            token_owners.setdefault(token.lower(), set()).add(alias)
    token_alias = {tok: next(iter(owners)) for tok, owners in token_owners.items() if len(owners) == 1}

    replacements: dict[str, str] = dict(full_alias)
    replacements.update(token_alias)
    next_extra = len(full_alias) + 1
    for variants in EXTRA_NAMES_TO_ANONYMIZE:
        alias = f"Third Party {next_extra:02d}"
        added = False
        for name in variants:
            if name.lower() not in {k.lower() for k in replacements}:
                replacements[name] = alias
                added = True
        if added:
            next_extra += 1

    # longest strings first + word boundaries, so "Ali" doesn't clip a
    # substring inside an unrelated longer word, and "Ali Ku" is matched
    # whole before its own "Ali" token would be
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in sorted(replacements, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )

    def _sub(m: "re.Match") -> str:
        matched = m.group(0)
        for key, alias in replacements.items():
            if key.lower() == matched.lower():
                return alias
        return matched  # pragma: no cover -- unreachable given how `pattern` is built

    for c in chunks:
        c.participants = [full_alias[p] for p in c.participants]
        c.text = pattern.sub(_sub, c.text)
    return chunks


def is_explicit(text: str) -> bool:
    lowered = f" {text.lower()} "
    return any(term in lowered for term in PROFANITY_TERMS)


def flag_explicit(chunks: List[Chunk]) -> List[Chunk]:
    for c in chunks:
        c.flagged_explicit = is_explicit(c.text)
    return chunks


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python ingest_whatsapp.py path/to/WhatsApp_Chat_export.txt")
    src = Path(sys.argv[1])
    out_dir = Path(__file__).resolve().parent

    messages = parse_export(src)
    chunks = to_chunks(messages)
    chunks = anonymize_names(chunks)
    chunks = flag_explicit(chunks)

    jsonl_path = out_dir / "study_chat_chunks.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "chunk_id": c.chunk_id,
                "start_ts": c.start_ts.isoformat(),
                "end_ts": c.end_ts.isoformat(),
                "participants": c.participants,
                "n_messages": c.n_messages,
                "still_flagged_pii": contains_pii(c.text),
                "flagged_explicit": c.flagged_explicit,
                "text": c.text,
            }, ensure_ascii=False) + "\n")

    kept = [c for c in chunks if not c.flagged_explicit]
    domain_knowledge_path = out_dir / "study_domain_knowledge.json"
    domain_knowledge_path.write_text(
        json.dumps([c.text for c in kept], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Parsed {len(messages)} raw messages")
    print(f"Produced {len(chunks)} chunks, {len(chunks) - len(kept)} filtered out (explicit content)")
    print(f"Kept {len(kept)} chunks in domain_knowledge")
    pii_left = sum(1 for c in kept if contains_pii(c.text))
    print(f"Kept chunks still flagged as containing PII after redaction: {pii_left}")
    print(f"Unique participants anonymized: {len({p for c in chunks for p in c.participants})}")
    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {domain_knowledge_path}")
    print(
        f"\nReminder: {src.name} is your raw, un-anonymized export -- do not commit it. "
        "Only the two files above are meant to ship. (.gitignore already excludes "
        "domains/*.txt and domains/WhatsApp* for this reason.) Do a manual skim of "
        f"{domain_knowledge_path.name} yourself before it goes into a public demo -- "
        "automated name-scrubbing on free text can't guarantee 100% recall."
    )


if __name__ == "__main__":
    main()
