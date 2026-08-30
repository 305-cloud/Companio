"""
belief.py - confidence as a calibrated Bayesian belief, not a magic float.

Every fact this companion holds about a person is a Beta(alpha, beta)
distribution: alpha counts confirming evidence, beta counts disconfirming
evidence. The posterior mean is "confidence"; alpha+beta ("concentration")
is how much evidence that number rests on -- a fact seen once and a fact
seen fifty times are no longer indistinguishable, which a single float
confidence can never express.

kl_divergence_beta gives a principled "surprise" measure: how much would
folding in one new observation move this belief? That question is what
memory/consolidator.py uses to decide whether a contradiction is real or
noise (see the module docstring there for why this replaced a fixed
`confidence >= 0.6` cutoff).

No scipy dependency -- lgamma is in the stdlib, and the digamma /
regularized-incomplete-beta implementations below are the standard
closed-form approximations, kept local so the whole companion package
stays install-free beyond the Python standard library.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass


def _digamma(x: float) -> float:
    """Digamma function via recurrence (push x >= 6) + asymptotic series."""
    result = 0.0
    while x < 6:
        result -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    result += (
        math.log(x) - 0.5 / x
        - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f * (1 / 132)))))
    )
    return result


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction evaluation used by the regularized incomplete beta."""
    maxit, eps, fpmin = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b), 0 <= x <= 1."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    ln_bt = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1 - x)
    )
    bt = math.exp(ln_bt)
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1 - x) / b


def _beta_ppf(p: float, a: float, b: float, iters: int = 60) -> float:
    """Inverse CDF of Beta(a, b) at probability p, via bisection on _betainc."""
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if _betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass
class BetaBelief:
    alpha: float = 1.0
    beta: float = 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def concentration(self) -> float:
        """How much evidence this belief rests on. Higher = more certain."""
        return self.alpha + self.beta

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def credible_interval(self, mass: float = 0.9) -> tuple[float, float]:
        lo_p = (1 - mass) / 2
        hi_p = 1 - lo_p
        return (
            _beta_ppf(lo_p, self.alpha, self.beta),
            _beta_ppf(hi_p, self.alpha, self.beta),
        )

    def update(self, agree: bool, weight: float = 1.0) -> "BetaBelief":
        """Fold in one piece of evidence. weight > 1 for stronger evidence."""
        if agree:
            self.alpha += weight
        else:
            self.beta += weight
        return self

    def decay(self, rho: float = 0.97) -> "BetaBelief":
        """Relax both parameters toward the uninformative prior (1, 1) --
        a genuine forgetting curve: confidence AND concentration both
        shrink without new evidence, with no risk of leaving [0, 1]."""
        self.alpha = 1 + (self.alpha - 1) * rho
        self.beta = 1 + (self.beta - 1) * rho
        return self

    def shift_surprise(self, agree: bool, weight: float = 1.0) -> float:
        """How much would folding in this one observation move the belief?
        KL(posterior-after || current), the informational cost of the
        update. This is what makes the surprise threshold automatically
        scale with evidence: nudging a belief backed by 100 observations
        by one more barely moves it (tiny KL); nudging a fresh belief by
        one observation moves it a lot (large KL) -- no separate
        evidence-scaling constant needed, unlike a fixed-threshold rule."""
        updated = copy.copy(self).update(agree, weight)
        return kl_divergence_beta(updated, self)

    def as_dict(self) -> dict:
        lo, hi = self.credible_interval()
        return {
            "confidence": round(self.mean, 3),
            "concentration": round(self.concentration, 2),
            "credible_interval_90": [round(lo, 3), round(hi, 3)],
        }


def kl_divergence_beta(p: BetaBelief, q: BetaBelief) -> float:
    """KL(P || Q) for two Beta distributions, closed form (in nats)."""
    a1, b1 = p.alpha, p.beta
    a2, b2 = q.alpha, q.beta
    betaln_p = math.lgamma(a1) + math.lgamma(b1) - math.lgamma(a1 + b1)
    betaln_q = math.lgamma(a2) + math.lgamma(b2) - math.lgamma(a2 + b2)
    return (
        (betaln_q - betaln_p)
        + (a1 - a2) * _digamma(a1)
        + (b1 - b2) * _digamma(b1)
        + (a2 - a1 + b2 - b1) * _digamma(a1 + b1)
    )
