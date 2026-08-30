import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from belief import BetaBelief, kl_divergence_beta  # noqa: E402
from gate import AdaptiveGate  # noqa: E402


def test_beta_belief_mean_and_bounds():
    b = BetaBelief()
    assert 0 <= b.mean <= 1
    b.update(agree=True, weight=5)
    assert b.mean > 0.5


def test_concentration_distinguishes_evidence_amount():
    seen_once = BetaBelief()
    seen_once.update(agree=True, weight=1)

    seen_fifty_times = BetaBelief()
    for _ in range(50):
        seen_fifty_times.update(agree=True, weight=1)

    assert seen_once.mean > 0.5 and seen_fifty_times.mean > 0.5
    assert seen_fifty_times.concentration > seen_once.concentration * 10


def test_decay_relaxes_toward_prior():
    b = BetaBelief(alpha=40, beta=2)
    before = b.mean
    for _ in range(20):
        b.decay(rho=0.9)
    after = b.mean
    assert after < before
    assert b.concentration < 42


def test_credible_interval_is_ordered_and_bounded():
    b = BetaBelief(alpha=10, beta=3)
    lo, hi = b.credible_interval(0.9)
    assert 0.0 <= lo < b.mean < hi <= 1.0


def test_kl_divergence_is_zero_for_identical_beliefs():
    a = BetaBelief(alpha=5, beta=5)
    b = BetaBelief(alpha=5, beta=5)
    assert kl_divergence_beta(a, b) < 1e-8


def test_shift_surprise_shrinks_as_evidence_accumulates():
    fresh = BetaBelief()
    fresh.update(agree=True, weight=1)

    established = BetaBelief()
    for _ in range(50):
        established.update(agree=True, weight=1)
    for _ in range(5):
        established.update(agree=False, weight=1)

    surprise_fresh = fresh.shift_surprise(agree=False, weight=1)
    surprise_established = established.shift_surprise(agree=False, weight=1)
    assert surprise_established < surprise_fresh


def test_shift_surprise_is_larger_for_a_never_challenged_belief():
    never_challenged = BetaBelief()
    for _ in range(50):
        never_challenged.update(agree=True, weight=1)

    modestly_confident = BetaBelief()
    modestly_confident.update(agree=True, weight=1)

    surprise_never_challenged = never_challenged.shift_surprise(agree=False, weight=1)
    surprise_modest = modestly_confident.shift_surprise(agree=False, weight=1)
    assert surprise_never_challenged > surprise_modest


def test_adaptive_gate_fires_on_deviation_not_absolute_level():
    gate = AdaptiveGate(alpha=0.2, delta=0.3)
    fired = []
    for value in [0.5, 0.5, 0.5, 0.5, 0.95]:
        f, _ = gate.step(value)
        fired.append(f)
    assert fired[-1] is True
    assert not any(fired[:-1])


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
