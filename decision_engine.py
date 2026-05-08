"""
Decision engine: generates candidate actions, scores them with predictions,
emotion modulation, metacognitive confidence, and personality traits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from emotion_engine import EmotionModulation
from ethics_engine import EthicsCheckResult, EthicsEngine
from metacognition_engine import MetacognitiveAssessment
from personality_engine import PersonalityProfile
from predictive_engine import PredictionBatch


@dataclass
class DecisionResult:
    """Committed decision with audit metadata."""

    decision_id: str
    selected_action: str
    confidence: float
    candidates: List[str]
    scores: List[float]
    ethics: EthicsCheckResult
    veto_reason: Optional[str]


class DecisionEngine:
    """Utility-based arbitration over a discrete action set."""

    def __init__(self, ethics: EthicsEngine) -> None:
        self._ethics = ethics
        self._seq = 0

    def propose_actions(
        self,
        *,
        recommend_more_sensing: bool,
        exploration: float,
    ) -> List[str]:
        """Produce candidate actions conditioned on metacognitive hints."""
        base = ["forward", "left", "right", "stay", "retreat"]
        if recommend_more_sensing:
            # Prefer slower / observational moves.
            return ["stay", "left", "right", "forward", "retreat"]
        if exploration > 0.55:
            # Slight reorder to prefer lateral exploration.
            return ["left", "right", "forward", "retreat", "stay"]
        return base

    def evaluate_actions(
        self,
        *,
        batch: PredictionBatch,
        emotion: EmotionModulation,
        meta: MetacognitiveAssessment,
        personality: PersonalityProfile,
        ethics_result: EthicsCheckResult,
    ) -> Tuple[List[float], List[str]]:
        """Compute adjusted scores; returns parallel lists to predictions order."""
        actions = [p.action for p in batch.predictions]
        scores: List[float] = []
        for p in batch.predictions:
            if not ethics_result.allowed and p.action.lower() not in ("stay", "hold", "stop", "s"):
                scores.append(-1e9)
                continue
            u = p.expected_utility
            # Penalize risk more when metacognitive confidence is low.
            u -= (1.1 + (1.0 - meta.confidence)) * p.risk
            # Personality: persistence reduces oscillation penalty for retreat.
            if p.action == "retreat":
                u -= 0.15 * personality.persistence
            # Exploration nudges lateral moves.
            if p.action in ("left", "right"):
                u += 0.12 * personality.exploration_bias * emotion.exploration_multiplier
            scores.append(u)
        return scores, actions

    def commit_action(
        self,
        *,
        actions: Sequence[str],
        scores: Sequence[float],
        ethics_result: EthicsCheckResult,
    ) -> DecisionResult:
        """Choose argmax unless ethics disallows; returns structured decision."""
        self._seq += 1
        decision_id = f"dec_{self._seq}"

        paired = sorted(zip(scores, actions), key=lambda x: x[0], reverse=True)
        veto_reason: Optional[str] = ethics_result.veto_reason

        selected: Optional[str] = None
        confidence = 0.0

        if not paired:
            selected = "stay"
            confidence = 0.1
        else:
            best_score, best_action = paired[0]
            second_score = paired[1][0] if len(paired) > 1 else best_score - 1.0
            margin = best_score - second_score
            confidence = float(max(0.0, min(1.0, 0.55 + 0.25 * margin)))

            if not ethics_result.allowed:
                # Force a safe primitive.
                selected = "stay"
                confidence = min(confidence, 0.35)
            else:
                selected = best_action

        return DecisionResult(
            decision_id=decision_id,
            selected_action=str(selected),
            confidence=confidence,
            candidates=list(actions),
            scores=list(scores),
            ethics=ethics_result,
            veto_reason=veto_reason if veto_reason and not ethics_result.allowed else None,
        )

    def result_to_dict(self, r: DecisionResult) -> Dict[str, Any]:
        return {
            "decisionId": r.decision_id,
            "selectedAction": r.selected_action,
            "confidence": r.confidence,
            "candidates": r.candidates,
            "scores": r.scores,
            "ethics": self._ethics.to_dict(r.ethics),
            "vetoReason": r.veto_reason,
        }
