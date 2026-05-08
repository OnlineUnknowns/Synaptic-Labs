"""
Predictive engine: short-horizon outcome estimates for discrete action candidates.

Uses an explicit lightweight model (no placeholders): distance-to-goal potential,
collision/threat penalty, energy proxy, and speed penalty vs ethics limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from emotion_engine import EmotionModulation
from self_model_engine import SelfState
from world_model_engine import WorldState


@dataclass
class ActionPrediction:
    """Predicted outcome metrics for one candidate action."""

    action: str
    expected_utility: float
    risk: float
    notes: Dict[str, float]


@dataclass
class PredictionBatch:
    """Batch of predictions for arbitration."""

    horizon_ms: int
    predictions: List[ActionPrediction]


class PredictiveEngine:
    """Score candidate motor commands before commitment."""

    def __init__(self, horizon_ms: int = 250) -> None:
        self._horizon_ms = horizon_ms

    def predict_batch(
        self,
        *,
        world: WorldState,
        self_state: SelfState,
        robot_position: Tuple[float, float, float],
        goal_position: Tuple[float, float, float],
        candidates: Sequence[str],
        emotion: EmotionModulation,
        ethics_speed_limit: float,
    ) -> PredictionBatch:
        """Return utility/risk estimates for each candidate."""
        preds: List[ActionPrediction] = []
        for action in candidates:
            eu, risk, notes = self._score_action(
                action=action,
                world=world,
                self_state=self_state,
                robot_position=robot_position,
                goal_position=goal_position,
                emotion=emotion,
                ethics_speed_limit=ethics_speed_limit,
            )
            preds.append(ActionPrediction(action=action, expected_utility=eu, risk=risk, notes=notes))
        return PredictionBatch(horizon_ms=self._horizon_ms, predictions=preds)

    def _score_action(
        self,
        *,
        action: str,
        world: WorldState,
        self_state: SelfState,
        robot_position: Tuple[float, float, float],
        goal_position: Tuple[float, float, float],
        emotion: EmotionModulation,
        ethics_speed_limit: float,
    ) -> Tuple[float, float, Dict[str, float]]:
        rx, ry, rz = robot_position
        gx, gy, gz = goal_position
        dist = max(1e-6, ((gx - rx) ** 2 + (gy - ry) ** 2 + (gz - rz) ** 2) ** 0.5)

        # Proposed motion step and implied speed proxy.
        dx, dy, dz = self._action_delta(action)
        step_len = max(1e-6, (dx**2 + dy**2 + dz**2) ** 0.5)
        speed_proxy = step_len / (self._horizon_ms / 1000.0)

        # Post-move position estimate (single-step Euler).
        nx, ny, nz = rx + dx, ry + dy, rz + dz

        # Goal progress reward: reduction in distance.
        new_dist = max(1e-6, ((gx - nx) ** 2 + (gy - ny) ** 2 + (gz - nz) ** 2) ** 0.5)
        progress = dist - new_dist

        # Threat: min distance to high-risk objects after move.
        min_hazard_dist = float("inf")
        max_risk = 0.0
        for obj in world.objects:
            if obj.risk <= 0.01:
                continue
            ox, oy, oz = obj.position
            d = ((ox - nx) ** 2 + (oy - ny) ** 2 + (oz - nz) ** 2) ** 0.5
            min_hazard_dist = min(min_hazard_dist, d)
            max_risk = max(max_risk, obj.risk)

        if min_hazard_dist == float("inf"):
            hazard_term = 0.0
            risk = max(0.0, min(1.0, world.uncertainty * 0.35))
        else:
            hazard_term = max(0.0, 1.0 - min_hazard_dist) * max_risk
            risk = max(0.0, min(1.0, 0.55 * hazard_term + 0.45 * world.uncertainty))

        # Energy and thermal pressure.
        energy_term = 0.12 * step_len + 0.25 * (1.0 - self_state.battery)
        thermal_term = 0.2 * self_state.thermal

        # Ethics: quadratic penalty beyond allowed speed.
        ethics_pen = 0.0
        if speed_proxy > ethics_speed_limit + 1e-6:
            over = speed_proxy - ethics_speed_limit
            ethics_pen = over**2

        # Compose utility (higher is better).
        utility = (
            1.4 * progress
            + 0.15 * (1.0 / new_dist)
            - emotion.risk_multiplier * (1.1 * hazard_term + 0.35 * risk)
            - 0.45 * energy_term
            - 0.35 * thermal_term
            - 0.9 * ethics_pen
            + 0.08 * emotion.exploration_multiplier * step_len
        )
        utility *= emotion.reward_gain

        notes = {
            "progress": progress,
            "new_distance": new_dist,
            "hazard_term": hazard_term,
            "speed_proxy": speed_proxy,
            "ethics_penalty": ethics_pen,
        }
        return utility, risk, notes

    @staticmethod
    def _action_delta(action: str) -> Tuple[float, float, float]:
        """Discrete action primitives in world units per prediction horizon."""
        a = action.lower().strip()
        step = 0.25
        if a in ("forward", "move_forward", "f"):
            return (step, 0.0, 0.0)
        if a in ("back", "backward", "retreat", "b"):
            return (-0.8 * step, 0.0, 0.0)
        if a in ("left", "turn_left", "l"):
            return (0.0, step, 0.0)
        if a in ("right", "turn_right", "r"):
            return (0.0, -step, 0.0)
        if a in ("stay", "hold", "stop", "s"):
            return (0.0, 0.0, 0.0)
        # default small exploratory move
        return (0.5 * step, 0.1 * step, 0.0)

    def batch_to_dict(self, batch: PredictionBatch) -> Dict[str, Any]:
        return {
            "horizonMs": batch.horizon_ms,
            "candidates": [p.action for p in batch.predictions],
            "expectedUtility": [p.expected_utility for p in batch.predictions],
            "risk": [p.risk for p in batch.predictions],
            "notes": [p.notes for p in batch.predictions],
        }
