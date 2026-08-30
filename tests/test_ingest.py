"""
Tests for domains/ingest_whatsapp.py's anonymization and safety-filter
logic, using small synthetic chat data (never the real uploaded dataset)
so this suite doesn't depend on -- or ship -- anyone's actual chat export.
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from domains.ingest_whatsapp import Chunk, anonymize_names, flag_explicit, is_explicit  # noqa: E402


def _chunk(chunk_id, participants, text):
    return Chunk(
        chunk_id=chunk_id, start_ts=datetime.now(), end_ts=datetime.now(),
        participants=participants, n_messages=len(participants), text=text,
    )


def test_anonymize_replaces_full_sender_name_consistently():
    chunks = [
        _chunk("c0", ["Amina Test"], "Amina Test: see you at 5"),
        _chunk("c1", ["Amina Test", "Ben Test"], "Amina Test: hey\nBen Test: hi Amina"),
    ]
    chunks = anonymize_names(chunks)
    full_text = "\n".join(c.text for c in chunks)
    assert "Amina" not in full_text
    assert "Ben" not in full_text
    # same person gets the same alias in both chunks
    assert chunks[0].text.split(":")[0] == chunks[1].text.split("\n")[0].split(":")[0]


def test_anonymize_catches_first_name_only_mentions():
    """The real bug this guards against: someone referred to by first name
    only inside a message body, not just in the 'Sender: text' prefix."""
    chunks = [_chunk("c0", ["Amina Test"], "Amina Test: reminder\nBen Test: thanks Amina, appreciate it")]
    chunks = anonymize_names(chunks)
    assert "Amina" not in chunks[0].text


def test_anonymize_handles_mentioned_but_never_sending_name():
    """A name that's only ever mentioned, never a sender, has nothing in
    `participants` to derive an alias from -- this documents that gap
    rather than silently leaking the name. See EXTRA_NAMES_TO_ANONYMIZE
    in ingest_whatsapp.py for the manual-supplement mechanism."""
    chunks = [_chunk("c0", ["Amina Test"], "Amina Test: ask Zara about it")]
    chunks = anonymize_names(chunks)
    # "Zara" isn't in participants anywhere, so it's NOT anonymized by
    # this function alone -- a known limitation, not a silent bug.
    assert "Zara" in chunks[0].text


def test_is_explicit_flags_profanity():
    assert is_explicit("this message contains lund and other bad words") is True
    assert is_explicit("let's meet at 5pm for the study session") is False


def test_flag_explicit_marks_chunks_without_dropping_them():
    chunks = [
        _chunk("c0", ["A"], "A: let's meet at the library"),
        _chunk("c1", ["A"], "A: you're being a chutiya about this"),
    ]
    chunks = flag_explicit(chunks)
    assert chunks[0].flagged_explicit is False
    assert chunks[1].flagged_explicit is True
    # flagging never deletes -- filtering to domain_knowledge is the
    # caller's job (main()), so both chunks are still present here
    assert len(chunks) == 2


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
