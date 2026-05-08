"""
Attention engine: fuses saliency, goal relevance, threat, novelty proxy, and
uncertainty into a bounded focus window with hysteresis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from personality_engine import PersonalityProfile
from world_model_engine import WorldObject, WorldState


@dataclass
class FocusItem:
    """One slot in the attention focus window."""

    item_id: str
    label: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FocusWindow:
    """Ranked attention context passed to prediction and decision."""

    items: List[FocusItem]
    capacity: int
    scores: List[float]


class AttentionEngine:
    """
    Computes priority scores and retains previous focus for hysteresis.

    Priority (higher is more important):
      w_sal * saliency
    + w_goal * goal_match
    + w_threat * risk
    + w_unc * world_uncertainty
    + w_nov * novelty_proxy
    + hysteresis_bonus if item was focused last cycle
    """

    def __init__(self, capacity: int = 8) -> None:
        self._capacity = max(1, capacity)
        self._last_focus_ids: List[str] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    def compute_focus(
        self,
        *,
        world: WorldState,
        goal_keywords: Sequence[str],
        personality: PersonalityProfile,
        memory_novelty_hint: float = 0.0,
    ) -> FocusWindow:
        """Build a prioritized focus window from world state and goals."""
        w_sal = 0.35
        w_goal = 0.35 + 0.15 * personality.persistence
        w_threat = 0.45 + 0.25 * (1.0 - personality.risk_tolerance)
        w_unc = 0.25
        w_nov = 0.15 + 0.2 * personality.exploration_bias

        items: List[FocusItem] = []
        goals = [g.lower() for g in goal_keywords]

        for obj in world.objects:
            goal_match = self._goal_match_score(obj, goals)
            novelty = max(0.0, min(1.0, memory_novelty_hint + (1.0 - obj.attributes.get("familiarity", 0.5))))
            score = (
                w_sal * obj.saliency
                + w_goal * goal_match
                + w_threat * obj.risk
                + w_unc * world.uncertainty
                + w_nov * novelty
            )
            if obj.object_id in self._last_focus_ids:
                score += 0.08  # hysteresis
            items.append(
                FocusItem(
                    item_id=obj.object_id,
                    label=obj.label,
                    score=score,
                    metadata={"risk": obj.risk, "position": obj.position},
                )
            )

        # Always include a synthetic ego slot if world is empty or low salience.
        if not items:
            items.append(
                FocusItem(
                    item_id="ego",
                    label="ego",
                    score=0.2 + w_unc * world.uncertainty,
                    metadata={},
                )
            )

        items.sort(key=lambda x: x.score, reverse=True)
        top = items[: self._capacity]
        self._last_focus_ids = [i.item_id for i in top]

        return FocusWindow(
            items=top,
            capacity=self._capacity,
            scores=[i.score for i in top],
        )

    @staticmethod
    def _goal_match_score(obj: WorldObject, goals: List[str]) -> float:
        if not goals:
            return 0.25
        label = obj.label.lower()
        best = 0.0
        for g in goals:
            if g in label or label in g:
                best = max(best, 1.0)
            elif any(tok in label for tok in g.split()):
                best = max(best, 0.65)
        return best

    def focus_to_dict(self, fw: FocusWindow) -> Dict[str, Any]:
        return {
            "focusItems": [i.item_id for i in fw.items],
            "capacity": fw.capacity,
            "scores": fw.scores,
            "details": [
                {"id": i.item_id, "label": i.label, "score": i.score, "metadata": i.metadata} for i in fw.items
            ],
        }
