"""
Synapse engine: eligibility traces, bounded Hebbian-style updates, homeostatic
normalization. Acts on abstract feature keys (string ids) for portability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


@dataclass
class PlasticityStats:
    """Summary statistics after an update step."""

    update_count: int
    l2_norm: float
    max_abs_weight: float


class SynapseEngine:
    """
    Maintains weights w[(pre, post)] and eligibility e[(pre, post)].

    Update rule on reward prediction error `delta`:
        e <- lambda * e + pre_activation * post_activation
        w <- clip(w + lr * delta * e)
        optional homeostatic scaling to keep L2 norm near target
    """

    def __init__(
        self,
        *,
        learning_rate: float,
        eligibility_decay: float,
        weight_clip: float,
        homeostatic_target_l2: float = 3.5,
        homeostatic_strength: float = 0.05,
    ) -> None:
        self._lr = float(learning_rate)
        self._lambda = float(eligibility_decay)
        self._clip = float(weight_clip)
        self._target_l2 = float(homeostatic_target_l2)
        self._homeo = float(homeostatic_strength)
        self._w: Dict[Tuple[str, str], float] = {}
        self._e: Dict[Tuple[str, str], float] = {}

    def ensure_connection(self, pre: str, post: str) -> None:
        key = (pre, post)
        self._w.setdefault(key, 0.1)
        self._e.setdefault(key, 0.0)

    def on_coactivation(self, pre: str, post: str, pre_act: float, post_act: float) -> None:
        """Call when pre/post features co-activate (e.g., decision-outcome pair)."""
        key = (pre, post)
        self._w.setdefault(key, 0.1)
        e_prev = self._e.get(key, 0.0)
        self._e[key] = self._lambda * e_prev + float(pre_act) * float(post_act)

    def apply_reward_error(self, delta: float) -> PlasticityStats:
        """Apply plasticity proportional to eligibility and prediction error."""
        update_count = 0
        for key, e in list(self._e.items()):
            dw = self._lr * float(delta) * e
            w0 = self._w.get(key, 0.1)
            w1 = max(-self._clip, min(self._clip, w0 + dw))
            self._w[key] = w1
            if abs(dw) > 1e-8:
                update_count += 1

        self._homeostatic_normalize()
        l2, mx = self._stats()
        # Decay eligibility after update to avoid unbounded credit assignment.
        for k in list(self._e.keys()):
            self._e[k] *= 0.65

        return PlasticityStats(update_count=update_count, l2_norm=l2, max_abs_weight=mx)

    def decay_eligibility(self) -> None:
        for k in list(self._e.keys()):
            self._e[k] *= self._lambda

    def _homeostatic_normalize(self) -> None:
        l2, _ = self._stats()
        if l2 < 1e-6:
            return
        factor = 1.0 - self._homeo * (1.0 - self._target_l2 / l2)
        factor = max(0.85, min(1.15, factor))
        for k in list(self._w.keys()):
            self._w[k] *= factor

    def _stats(self) -> Tuple[float, float]:
        sq = sum(v * v for v in self._w.values())
        l2 = sq**0.5
        mx = max((abs(v) for v in self._w.values()), default=0.0)
        return l2, mx

    def weight_norms(self) -> Tuple[float, float]:
        """Public L2 norm and max absolute weight (for monitoring/consolidation)."""
        return self._stats()

    def read_weight(self, pre: str, post: str) -> float:
        return self._w.get((pre, post), 0.0)

    def export_weights(self) -> Mapping[Tuple[str, str], float]:
        return dict(self._w)

    def import_weights(self, weights: Mapping[Tuple[str, str], float]) -> None:
        self._w = {k: float(v) for k, v in weights.items()}
        self._e = {}

    def normalize_full(self) -> None:
        """Sleep consolidation helper: rescale all weights to target L2."""
        l2, _ = self._stats()
        if l2 < 1e-6:
            return
        scale = self._target_l2 / l2
        for k in list(self._w.keys()):
            self._w[k] = max(-self._clip, min(self._clip, self._w[k] * scale))
