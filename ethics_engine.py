"""
Ethics engine: policy constraints and hard vetoes on forbidden actions.

Implements explicit rules (production pattern: load policies from DB/config).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set


@dataclass
class EthicsCheckResult:
    """Outcome of an ethics evaluation."""

    allowed: bool
    forbidden_hits: List[str]
    speed_limit: float
    constraints: Dict[str, Any]
    veto_reason: Optional[str] = None


class EthicsEngine:
    """
    Evaluates candidate actions against forbidden labels and speed limits.

    `forbidden_object_labels` blocks actions whose predicted focus targets
    match sensitive categories (e.g., human, fragile).
    """

    def __init__(self, max_speed: float, forbidden_labels: Optional[Sequence[str]] = None) -> None:
        self._max_speed = max(0.0, float(max_speed))
        self._forbidden: Set[str] = {s.lower() for s in (forbidden_labels or ("human", "child", "pet"))}

    @property
    def speed_limit(self) -> float:
        return self._max_speed

    def evaluate_candidates(
        self,
        *,
        candidates: Sequence[str],
        focus_labels: Sequence[str],
        proposed_speed: float,
    ) -> EthicsCheckResult:
        """Return whether candidates pass policy; may veto entire batch."""
        forbidden_hits: List[str] = []
        lowered_focus = [f.lower() for f in focus_labels]

        for label in lowered_focus:
            if any(f in label for f in self._forbidden):
                forbidden_hits.append(label)

        # If current attention is dominated by forbidden entity, block aggressive motion.
        if forbidden_hits:
            allowed = all(c.lower() in ("stay", "hold", "stop", "s") for c in candidates)
            return EthicsCheckResult(
                allowed=allowed,
                forbidden_hits=forbidden_hits,
                speed_limit=min(self._max_speed, 0.15),
                constraints={"mode": "restricted_proximity"},
                veto_reason="forbidden_entity_in_focus" if not allowed else None,
            )

        if proposed_speed > self._max_speed + 1e-6:
            return EthicsCheckResult(
                allowed=False,
                forbidden_hits=[],
                speed_limit=self._max_speed,
                constraints={"mode": "speed_violation"},
                veto_reason="speed_limit_exceeded",
            )

        return EthicsCheckResult(
            allowed=True,
            forbidden_hits=[],
            speed_limit=self._max_speed,
            constraints={"mode": "nominal"},
            veto_reason=None,
        )

    def evaluate_action_with_context(
        self,
        *,
        action: str,
        focus_labels: Sequence[str],
        speed_proxy: float,
    ) -> EthicsCheckResult:
        """Convenience single-action check used before motor commit."""
        return self.evaluate_candidates(
            candidates=[action],
            focus_labels=focus_labels,
            proposed_speed=speed_proxy,
        )

    def to_dict(self, r: EthicsCheckResult) -> Dict[str, Any]:
        return {
            "allowed": r.allowed,
            "forbiddenHits": r.forbidden_hits,
            "speedLimit": r.speed_limit,
            "constraints": r.constraints,
            "vetoReason": r.veto_reason,
        }
