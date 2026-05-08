"""
Emergency engine: deterministic hazard detection and motor preemption.

Runs authoritative checks on proximity, thermal stress, actuator health, and
localization trust. Can force safe-stop/hold independent of learned policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from self_model_engine import SelfState
from world_model_engine import WorldModelEngine


class EmergencyLevel(str, Enum):
    NONE = "none"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EmergencyStatus:
    """Current emergency evaluation."""

    level: EmergencyLevel
    reason: str
    safe_action: str
    hazard_class: str


class EmergencyEngine:
    """
    Produces emergency status and overrides for the motor engine.

    Parameters `distance_threshold` and thermal limits align with config.
    """

    def __init__(
        self,
        *,
        distance_threshold: float,
        thermal_critical: float = 0.92,
        battery_critical: float = 0.08,
        health_critical: float = 0.35,
    ) -> None:
        self._distance_threshold = float(distance_threshold)
        self._thermal_critical = float(thermal_critical)
        self._battery_critical = float(battery_critical)
        self._health_critical = float(health_critical)

    def evaluate(
        self,
        *,
        world: WorldModelEngine,
        self_state: SelfState,
        robot_position: Tuple[float, float, float],
    ) -> EmergencyStatus:
        """Classify hazard given world/self estimates."""
        d, obj = world.nearest_hazard(robot_position)
        if d < self._distance_threshold and obj is not None:
            return EmergencyStatus(
                level=EmergencyLevel.CRITICAL,
                reason=f"proximity_hazard:{obj.object_id}",
                safe_action="stop",
                hazard_class="collision_imminent",
            )

        if self_state.thermal >= self._thermal_critical:
            return EmergencyStatus(
                level=EmergencyLevel.HIGH,
                reason="thermal_critical",
                safe_action="hold",
                hazard_class="thermal_overload",
            )

        if self_state.battery <= self._battery_critical:
            return EmergencyStatus(
                level=EmergencyLevel.HIGH,
                reason="battery_critical",
                safe_action="hold",
                hazard_class="power_critical",
            )

        if self_state.actuator_health <= self._health_critical:
            return EmergencyStatus(
                level=EmergencyLevel.HIGH,
                reason="actuator_health_critical",
                safe_action="hold",
                hazard_class="actuator_fault",
            )

        if self_state.localization_trust <= 0.25:
            return EmergencyStatus(
                level=EmergencyLevel.WARNING,
                reason="localization_untrusted",
                safe_action="hold",
                hazard_class="localization_loss_critical",
            )

        return EmergencyStatus(
            level=EmergencyLevel.NONE,
            reason="ok",
            safe_action="nominal",
            hazard_class="none",
        )

    def should_veto_motion(self, status: EmergencyStatus) -> bool:
        return status.level in (EmergencyLevel.CRITICAL, EmergencyLevel.HIGH)

    def to_dict(self, status: EmergencyStatus) -> Dict[str, Any]:
        return {
            "level": status.level.value,
            "reason": status.reason,
            "safeAction": status.safe_action,
            "hazardClass": status.hazard_class,
        }
