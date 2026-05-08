"""
Metacognition engine: tracks recent prediction errors and emits confidence and
strategy hints (e.g., request more sensing, prefer conservative actions).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Deque, Dict, List


@dataclass
class MetacognitiveAssessment:
    """Calibration-style assessment for decision biasing."""

    confidence: float
    prediction_error_mean: float
    prediction_error_std: float
    recommend_replan: bool
    recommend_more_sensing: bool
    rationale: str


class MetacognitionEngine:
    """Online monitoring of model divergence via reward prediction errors."""

    def __init__(self, window: int = 32) -> None:
        self._window: int = max(8, window)
        self._errors: Deque[float] = deque(maxlen=self._window)

    def observe_prediction_error(self, prediction_error: float) -> None:
        self._errors.append(float(prediction_error))

    def assess(
        self,
        *,
        world_uncertainty: float,
        self_confidence: float,
    ) -> MetacognitiveAssessment:
        """Produce an assessment from recent errors and state uncertainty."""
        if len(self._errors) < 5:
            err_mean = float(mean(self._errors)) if self._errors else 0.0
            err_std = float(pstdev(self._errors)) if len(self._errors) > 1 else 0.0
            base_conf = 0.75 * (1.0 - world_uncertainty) + 0.25 * self_confidence
            return MetacognitiveAssessment(
                confidence=max(0.0, min(1.0, base_conf)),
                prediction_error_mean=err_mean,
                prediction_error_std=err_std,
                recommend_replan=False,
                recommend_more_sensing=world_uncertainty > 0.55,
                rationale="warming_up",
            )

        err_mean = float(mean(self._errors))
        err_std = float(pstdev(self._errors))

        # Confidence decreases with high average error and high volatility.
        err_penalty = min(1.0, abs(err_mean) + 0.75 * err_std)
        confidence = (
            0.55 * (1.0 - world_uncertainty)
            + 0.35 * self_confidence
            + 0.1 * max(0.0, 1.0 - err_penalty)
        )
        confidence = max(0.0, min(1.0, confidence))

        recommend_replan = err_std > 0.35 or abs(err_mean) > 0.45
        recommend_more_sensing = world_uncertainty > 0.45 or err_std > 0.3

        rationale = "stable"
        if recommend_replan:
            rationale = "high_error_volatility"
        elif recommend_more_sensing:
            rationale = "perception_uncertainty"

        return MetacognitiveAssessment(
            confidence=confidence,
            prediction_error_mean=err_mean,
            prediction_error_std=err_std,
            recommend_replan=recommend_replan,
            recommend_more_sensing=recommend_more_sensing,
            rationale=rationale,
        )

    def to_dict(self, a: MetacognitiveAssessment) -> Dict[str, Any]:
        return {
            "confidence": a.confidence,
            "predictionErrorMean": a.prediction_error_mean,
            "predictionErrorStd": a.prediction_error_std,
            "recommendReplan": a.recommend_replan,
            "recommendMoreSensing": a.recommend_more_sensing,
            "rationale": a.rationale,
        }

    def recent_errors(self) -> List[float]:
        return list(self._errors)
