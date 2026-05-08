"""
Personality engine: stable trait priors that bias exploration, risk, and persistence.

Traits are loaded once and exposed to attention and decision engines as multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class PersonalityProfile:
    """Immutable personality parameters in normalized ranges."""

    risk_tolerance: float
    exploration_bias: float
    persistence: float
    social_cooperation: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_tolerance": self.risk_tolerance,
            "exploration_bias": self.exploration_bias,
            "persistence": self.persistence,
            "social_cooperation": self.social_cooperation,
        }


class PersonalityEngine:
    """Provides stable behavioral priors."""

    def __init__(self, profile: PersonalityProfile | None = None) -> None:
        self._profile = profile or PersonalityProfile(
            risk_tolerance=0.55,
            exploration_bias=0.45,
            persistence=0.6,
            social_cooperation=0.7,
        )

    @property
    def profile(self) -> PersonalityProfile:
        return self._profile

    def load_from_dict(self, data: Dict[str, Any]) -> PersonalityProfile:
        """Replace profile from a dictionary (all keys optional)."""
        p = self._profile
        new = PersonalityProfile(
            risk_tolerance=float(data.get("risk_tolerance", p.risk_tolerance)),
            exploration_bias=float(data.get("exploration_bias", p.exploration_bias)),
            persistence=float(data.get("persistence", p.persistence)),
            social_cooperation=float(data.get("social_cooperation", p.social_cooperation)),
        )
        self._profile = self._clamp(new)
        return self._profile

    @staticmethod
    def _clamp(p: PersonalityProfile) -> PersonalityProfile:
        def c(x: float) -> float:
            return max(0.0, min(1.0, x))

        return PersonalityProfile(
            risk_tolerance=c(p.risk_tolerance),
            exploration_bias=c(p.exploration_bias),
            persistence=c(p.persistence),
            social_cooperation=c(p.social_cooperation),
        )
