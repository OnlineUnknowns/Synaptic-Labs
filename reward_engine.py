"""
Reward engine: composes extrinsic task signals with intrinsic shaping and
computes temporal-difference style prediction error for learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple


@dataclass
class RewardBreakdown:
    """Decomposed reward for audit and visualization."""

    total: float
    task: float
    safety: float
    energy: float
    novelty: float
    penalty: float


class RewardEngine:
    """
    Maintains a simple value estimate V for baseline subtraction (TD error).

    reward = task_progress + safety_bonus - energy_cost + novelty - penalties
    prediction_error = reward - (V_prev - gamma * V_next_approx)
    """

    def __init__(self, *, gamma: float = 0.95, value_lr: float = 0.15) -> None:
        self._gamma = float(gamma)
        self._value_lr = float(value_lr)
        self._value = 0.0

    @property
    def value_estimate(self) -> float:
        return float(self._value)

    def compute_reward(
        self,
        *,
        task_progress: float,
        hazard_proximity: float,
        battery: float,
        novelty: float,
        ethics_violation: bool,
        emergency_level: str,
    ) -> Tuple[float, RewardBreakdown, float]:
        """
        Compute scalar reward, breakdown, and prediction error.

        `hazard_proximity` in [0,1] (closer to hazard -> higher).
        """
        task = 1.8 * float(task_progress)
        safety = 0.55 * (1.0 - max(0.0, min(1.0, hazard_proximity)))
        energy = -0.35 * (1.0 - max(0.0, min(1.0, battery))) ** 1.25
        novelty = 0.12 * max(0.0, min(1.0, novelty))
        penalty = 0.0
        if ethics_violation:
            penalty -= 1.25
        if emergency_level in ("high", "critical"):
            penalty -= 1.0 if emergency_level == "high" else 2.0

        total = task + safety + energy + novelty + penalty
        total = float(max(-5.0, min(5.0, total)))

        breakdown = RewardBreakdown(
            total=total,
            task=task,
            safety=safety,
            energy=energy,
            novelty=novelty,
            penalty=penalty,
        )

        # TD error proxy: compare reward to current value baseline.
        pred_err = total - self._value
        self._value = self._value + self._value_lr * pred_err
        return total, breakdown, float(pred_err)

    def apply_credit_assignment(
        self,
        *,
        prediction_error: float,
        action_name: str,
        focus_token: str,
        synapse_on_coactivation: Callable[[str, str, float, float], None],
    ) -> Dict[str, Any]:
        """
        Bridge to synaptic updates: strengthens action-outcome associations.

        `synapse_on_coactivation` is expected to be `SynapseEngine.on_coactivation`.
        """
        pre = f"act:{action_name}"
        post = f"ctx:{focus_token}"
        synapse_on_coactivation(pre, post, pre_act=1.0, post_act=1.0)
        return {"pre": pre, "post": post, "predictionError": prediction_error}
