"""
Self model: internal robot state — power, thermal, actuator health, load.

Consumes telemetry frames and motor outcomes to update confidence in the
current operating envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class SelfState:
    """Robot introspective state used by attention, emergency, and decision."""

    battery: float
    thermal: float
    cpu_load: float
    actuator_health: float
    localization_trust: float
    confidence: float
    extras: Dict[str, Any] = field(default_factory=dict)


class SelfModelEngine:
    """
    Maintains `SelfState` from telemetry.

    Telemetry frame example:
        {
          "battery": 0.82,
          "thermal_c": 55.0,
          "cpu_load": 0.35,
          "actuator_health": 0.95,
          "localization_trust": 0.9
        }
    """

    def __init__(self) -> None:
        self._state = SelfState(
            battery=1.0,
            thermal=0.35,
            cpu_load=0.1,
            actuator_health=1.0,
            localization_trust=0.9,
            confidence=1.0,
        )

    @property
    def state(self) -> SelfState:
        return self._state

    def ingest_telemetry(self, telemetry: Dict[str, Any]) -> SelfState:
        """Update internal estimates from a telemetry dictionary."""
        bat = float(telemetry.get("battery", self._state.battery))
        thermal_c = float(telemetry.get("thermal_c", 45.0))
        cpu = float(telemetry.get("cpu_load", self._state.cpu_load))
        health = float(telemetry.get("actuator_health", self._state.actuator_health))
        loc_trust = float(telemetry.get("localization_trust", self._state.localization_trust))

        bat = max(0.0, min(1.0, bat))
        cpu = max(0.0, min(1.0, cpu))
        health = max(0.0, min(1.0, health))
        loc_trust = max(0.0, min(1.0, loc_trust))

        # Normalize thermal to 0..1 with 85C as high stress.
        thermal_01 = max(0.0, min(1.0, thermal_c / 85.0))

        # Confidence combines resource and health headroom.
        confidence = (
            0.35 * bat
            + 0.25 * (1.0 - thermal_01)
            + 0.2 * health
            + 0.2 * loc_trust
        )
        confidence = max(0.0, min(1.0, confidence))

        self._state = SelfState(
            battery=bat,
            thermal=thermal_01,
            cpu_load=cpu,
            actuator_health=health,
            localization_trust=loc_trust,
            confidence=confidence,
            extras={k: v for k, v in telemetry.items() if k not in {"battery", "thermal_c", "cpu_load"}},
        )
        return self._state

    def register_motor_stress(self, command_magnitude: float) -> None:
        """Increase CPU/load proxy slightly based on issued motor command intensity."""
        delta = min(0.15, 0.02 + 0.05 * max(0.0, command_magnitude))
        cpu = min(1.0, self._state.cpu_load + delta)
        thermal = min(1.0, self._state.thermal + 0.01 * max(0.0, command_magnitude))
        self._state = SelfState(
            battery=self._state.battery,
            thermal=thermal,
            cpu_load=cpu,
            actuator_health=self._state.actuator_health,
            localization_trust=self._state.localization_trust,
            confidence=max(0.0, self._state.confidence - 0.01 * command_magnitude),
            extras=dict(self._state.extras),
        )

    @staticmethod
    def default_robot_position() -> Tuple[float, float, float]:
        """Placeholder ego position when not provided by localization (simulator origin)."""
        return (0.0, 0.0, 0.0)
