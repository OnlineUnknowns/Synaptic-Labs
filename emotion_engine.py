"""
Emotion engine: low-dimensional affect (valence, arousal, control) with decay.

Produces multipliers that modulate risk sensitivity and exploration in the
decision engine. Designed to be deterministic given the same inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class EmotionState:
    """Circumplex-style affect state, each dimension in [0, 1]."""

    valence: float
    arousal: float
    control: float

    def to_dict(self) -> Dict[str, float]:
        return {"valence": self.valence, "arousal": self.arousal, "control": self.control}


@dataclass
class EmotionModulation:
    """Coefficients applied to decision utility components."""

    risk_multiplier: float
    exploration_multiplier: float
    reward_gain: float


class EmotionEngine:
    """
    Updates affect from rewards and hazard proximity, applies exponential decay
    toward baseline each step.
    """

    def __init__(
        self,
        baseline: Tuple[float, float, float] = (0.55, 0.25, 0.7),
        decay: float = 0.85,
    ) -> None:
        self._decay = decay
        self._baseline = EmotionState(*baseline)
        self._state = EmotionState(*baseline)

    @property
    def state(self) -> EmotionState:
        return self._state

    def reset(self) -> None:
        self._state = EmotionState(self._baseline.valence, self._baseline.arousal, self._baseline.control)

    def step_decay(self) -> EmotionState:
        """Pull affect toward baseline (call once per cognitive cycle)."""
        v = self._decay * self._state.valence + (1 - self._decay) * self._baseline.valence
        a = self._decay * self._state.arousal + (1 - self._decay) * self._baseline.arousal
        c = self._decay * self._state.control + (1 - self._decay) * self._baseline.control
        self._state = EmotionState(
            valence=max(0.0, min(1.0, v)),
            arousal=max(0.0, min(1.0, a)),
            control=max(0.0, min(1.0, c)),
        )
        return self._state

    def apply_reward(self, reward_value: float) -> None:
        """Positive rewards increase valence; negative rewards decrease it."""
        delta_v = 0.08 * max(-1.0, min(1.0, reward_value))
        delta_a = 0.04 * abs(reward_value)
        self._state = EmotionState(
            valence=max(0.0, min(1.0, self._state.valence + delta_v)),
            arousal=max(0.0, min(1.0, self._state.arousal + delta_a)),
            control=max(0.0, min(1.0, self._state.control + 0.01 * reward_value)),
        )

    def apply_hazard(self, hazard_severity: float) -> None:
        """Hazard increases arousal and reduces valence/control."""
        sev = max(0.0, min(1.0, hazard_severity))
        self._state = EmotionState(
            valence=max(0.0, min(1.0, self._state.valence - 0.15 * sev)),
            arousal=max(0.0, min(1.0, self._state.arousal + 0.25 * sev)),
            control=max(0.0, min(1.0, self._state.control - 0.1 * sev)),
        )

    def compute_modulation(self, personality_risk_tolerance: float) -> EmotionModulation:
        """
        Map emotion + personality into decision multipliers.

        Higher arousal and lower valence increase risk penalty. Higher control
        restores exploration slightly.
        """
        v, a, c = self._state.valence, self._state.arousal, self._state.control
        # Risk multiplier: >1 means more conservative (utility penalty scaled up).
        risk_multiplier = 1.0 + (0.6 * a) + (0.4 * (1.0 - v)) - (0.2 * personality_risk_tolerance)
        risk_multiplier = max(0.5, min(2.5, risk_multiplier))

        exploration_multiplier = (
            0.55 * c + 0.25 * v + 0.2 * personality_risk_tolerance - 0.35 * a
        )
        exploration_multiplier = max(0.2, min(1.5, exploration_multiplier))

        reward_gain = 0.85 + 0.3 * (v - 0.5)
        reward_gain = max(0.5, min(1.25, reward_gain))

        return EmotionModulation(
            risk_multiplier=risk_multiplier,
            exploration_multiplier=exploration_multiplier,
            reward_gain=reward_gain,
        )

    def to_audit_dict(self) -> Dict[str, Any]:
        return {"emotion": self._state.to_dict()}
