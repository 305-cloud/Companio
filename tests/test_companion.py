import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent import Companion  # noqa: E402
from domain import DomainConfig  # noqa: E402


def make_domain():
    return DomainConfig(
        name="test_domain",
        purpose="Test purpose",
        required_slots=["category"],
        clarifying_question_bank=["Which category is this?"],
        domain_knowledge=[
            "Test knowledge snippet about routines and category items.",
            "A second snippet, also about category routines.",
        ],
    )


def test_first_turn_asks_clarifying_question_when_context_is_thin():
    c = Companion(domain=make_domain())
    result = c.turn("u1", "hey")
    assert result.asked_clarifying is True
    assert result.confidence < 0.4


def test_turn_with_slot_filled_proceeds():
    c = Companion(domain=make_domain())
    result = c.turn("u1", "this is a category question about routines")
    assert result.asked_clarifying is False
    assert isinstance(result.response, str) and len(result.response) > 0


def test_feedback_and_consolidation_creates_semantic_fact():
    c = Companion(domain=make_domain())
    c.turn("u1", "category routines question")
    c.give_feedback("u1", "correction", {
        "key": "pref_style", "label": "Prefers concise answers", "value": True, "confidence": 1.0,
    })
    c.consolidate("u1")
    profile = c.profile("u1")
    assert any(f["key"] == "pref_style" for f in profile)


def test_reinforced_fact_gains_concentration():
    c = Companion(domain=make_domain())
    for _ in range(5):
        c.give_feedback("u1", "correction", {"key": "k", "label": "A", "value": "A", "confidence": 1.0})
        c.consolidate("u1")
    fact = c._load_internal_state("u1").get("k")
    assert fact.belief is not None
    assert fact.belief.concentration > 5


def test_strong_contradiction_marks_pending_not_silent_overwrite():
    c = Companion(domain=make_domain())
    for _ in range(10):
        c.give_feedback("u1", "correction", {"key": "k", "label": "A", "value": "A", "confidence": 1.0})
        c.consolidate("u1")
    c.give_feedback("u1", "correction", {"key": "k", "label": "B", "value": "B", "confidence": 1.0})
    c.consolidate("u1")
    fact = c._load_internal_state("u1").get("k")
    assert fact.status == "pending_confirmation"
    assert fact.value == "A"
    assert fact.pending_value == "B"


def test_resolve_pending_accept_applies_staged_value():
    c = Companion(domain=make_domain())
    for _ in range(10):
        c.give_feedback("u1", "correction", {"key": "k", "label": "A", "value": "A", "confidence": 1.0})
        c.consolidate("u1")
    c.give_feedback("u1", "correction", {"key": "k", "label": "B", "value": "B", "confidence": 1.0})
    c.consolidate("u1")
    c.resolve_pending("u1", "k", accept_update=True)
    fact = c._load_internal_state("u1").get("k")
    assert fact.status == "active"
    assert fact.value == "B"


def test_forget_removes_fact():
    c = Companion(domain=make_domain())
    c.give_feedback("u1", "correction", {"key": "k", "label": "A", "value": "A", "confidence": 1.0})
    c.consolidate("u1")
    assert c.forget("u1", "k") is True
    assert c._load_internal_state("u1").get("k") is None


def test_pii_is_flagged_in_episodic_log():
    c = Companion(domain=make_domain())
    c.turn("u1", "my email is test@example.com and category is routines")
    events = c.store.read_episodic("u1")
    assert any(e.pii for e in events if e.event_type == "input")


def test_idempotent_write_does_not_duplicate():
    from memory.store import EpisodicEvent, SOURCE_HUMAN, UnifiedMemoryStore
    store = UnifiedMemoryStore()
    event = EpisodicEvent.new("u1", SOURCE_HUMAN, "input", {"text": "hi"})
    store.write_episodic(event)
    store.write_episodic(event)
    rows = store.read_episodic("u1")
    assert len(rows) == 1


def test_metrics_track_session_trend():
    c = Companion(domain=make_domain())
    c.turn("u1", "hey")
    c.turn("u1", "category routines")
    trend = c.adaptation_metrics("u1")
    assert trend[0]["turns"] == 2


def test_feedback_closes_clarifier_learning_loop():
    c = Companion(domain=make_domain())
    c.turn("u1", "hey")
    features, _ = c._last_decision["u1"]
    p_before = c.clarifier._score(features)
    c.give_feedback("u1", "down")
    p_after = c.clarifier._score(features)
    assert p_after != p_before


def test_correction_feedback_does_not_touch_clarifier_weights():
    """A 'correction' teaches the memory system a fact -- it isn't a
    judgment on whether the previous turn was right to ask or not, so it
    shouldn't push a reward into the Clarifier the way 'up'/'down' do."""
    c = Companion(domain=make_domain())
    c.turn("u1", "hey")
    features, _ = c._last_decision["u1"]
    p_before = c.clarifier._score(features)
    c.give_feedback("u1", "correction", {"key": "k", "label": "A", "value": "A", "confidence": 1.0})
    p_after = c.clarifier._score(features)
    assert p_after == p_before


def test_relevant_profile_fact_is_not_swallowed_by_entropy_override():
    """Binary entropy is very flat near its peak: entropy(0.44) is ~0.99,
    almost as high as entropy(0.50)'s max of 1.0. With too-low a
    tau_entropy, the entropy check silently overrides a perfectly good
    mid-confidence PROCEED_WITH_FLAG decision back to ASK -- discarding
    a profile fact that was already retrieved and relevant, even though
    the raw confidence score alone would have said "proceed". This
    reproduces that exact scenario and pins the fix (tau_entropy raised
    from 0.95 to 0.99, so only genuinely ~50/50 confidence forces a
    question)."""
    c = Companion(domain=make_domain())
    c.turn("u1", "hey")
    c.give_feedback("u1", "correction", {"key": "k", "label": "A", "value": "A", "confidence": 1.0})
    c.consolidate("u1")

    result = c.turn("u1", "tell me about routines")

    assert result.asked_clarifying is False
    assert result.used_profile is True


def test_knowledgeless_domain_does_not_get_stuck_asking_forever():
    """A domain with no domain_knowledge at all (like GENERAL_DOMAIN) used
    to fall back to retrieval_score=0.5 -- meant as "neutral", but read by
    the Clarifier as "maximally uncertain" (entropy peaks at p=0.5). With
    empty required_slots too, every turn produced the exact same features
    and therefore the exact same confidence, permanently trapped in the
    entropy-forces-ASK zone: a fresh user got the identical clarifying
    question no matter what they typed, forever, since profile_match_strength
    never has a chance to grow without a first real response ever landing.
    retrieval_score now defaults to 1.0 for a domain with no knowledge base
    (nothing to fall short of), which moves confidence out of the trap."""
    from domain import DomainConfig

    knowledgeless = DomainConfig(
        name="empty_domain", purpose="No fixed vertical, no knowledge base.",
        required_slots=[], clarifying_question_bank=["Tell me more?"],
        domain_knowledge=[],
    )
    c = Companion(domain=knowledgeless)
    for text in ["hey", "i need advice", "how are you", "what's the capital of France"]:
        result = c.turn("u1", text)
        assert result.asked_clarifying is False


def test_general_and_study_domains_both_run():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from domains.general import GENERAL_DOMAIN
    from domains.study import STUDY_DOMAIN

    for domain in (GENERAL_DOMAIN, STUDY_DOMAIN):
        c = Companion(domain=domain)
        result = c.turn("u1", "hello there")
        assert isinstance(result.response, str)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
