"""
World model: maintains a lightweight scene representation with uncertainty.

Updates from sensor frames (simulated or robot-derived). Objects carry position,
semantic type, and optional risk score for attention and emergency engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class WorldObject:
    """Single tracked entity in the robot's environment."""

    object_id: str
    label: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    saliency: float = 0.5
    risk: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldState:
    """Aggregated world state published to downstream engines."""

    world_state_id: str
    objects: List[WorldObject]
    uncertainty: float
    affordances: Dict[str, float] = field(default_factory=dict)
    timestamp_ms: int = 0


class WorldModelEngine:
    """
    Maintains world state from structured sensor payloads.

    Expected sensor frame shape (example):
        {
          "timestamp_ms": 1234567890,
          "objects": [
            {"id": "obs1", "label": "obstacle", "position": [1.0,0.0,0.0], "risk": 0.8}
          ],
          "localization_confidence": 0.92
        }
    """

    def __init__(self) -> None:
        self._state: Optional[WorldState] = None
        self._seq = 0

    @property
    def state(self) -> Optional[WorldState]:
        return self._state

    def ingest_sensor_frame(self, frame: Dict[str, Any]) -> WorldState:
        """Parse sensor frame dict into `WorldState` and store."""
        ts = int(frame.get("timestamp_ms", 0))
        loc_conf = float(frame.get("localization_confidence", 0.75))
        loc_conf = max(0.0, min(1.0, loc_conf))

        objects: List[WorldObject] = []
        for raw in frame.get("objects", []) or []:
            oid = str(raw.get("id", f"obj_{len(objects)}"))
            label = str(raw.get("label", "unknown"))
            pos = raw.get("position", [0.0, 0.0, 0.0])
            if len(pos) < 3:
                pos = list(pos) + [0.0] * (3 - len(pos))
            vel = raw.get("velocity", [0.0, 0.0, 0.0])
            if len(vel) < 3:
                vel = list(vel) + [0.0] * (3 - len(vel))
            saliency = float(raw.get("saliency", 0.5))
            risk = float(raw.get("risk", 0.0))
            objects.append(
                WorldObject(
                    object_id=oid,
                    label=label,
                    position=(float(pos[0]), float(pos[1]), float(pos[2])),
                    velocity=(float(vel[0]), float(vel[1]), float(vel[2])),
                    saliency=max(0.0, min(1.0, saliency)),
                    risk=max(0.0, min(1.0, risk)),
                    attributes=dict(raw.get("attributes", {})),
                )
            )

        # Uncertainty rises when localization confidence drops.
        uncertainty = max(0.0, min(1.0, 1.0 - loc_conf))
        self._seq += 1
        world_state_id = f"ws_{self._seq}"

        affordances = dict(frame.get("affordances", {}))
        self._state = WorldState(
            world_state_id=world_state_id,
            objects=objects,
            uncertainty=uncertainty,
            affordances=affordances,
            timestamp_ms=ts,
        )
        return self._state

    def nearest_hazard(self, robot_position: Tuple[float, float, float]) -> Tuple[float, Optional[WorldObject]]:
        """Return (distance, object) to the highest-risk nearby object."""
        if not self._state or not self._state.objects:
            return float("inf"), None
        best: Optional[WorldObject] = None
        best_d = float("inf")
        rx, ry, rz = robot_position
        for obj in self._state.objects:
            if obj.risk <= 0.01:
                continue
            ox, oy, oz = obj.position
            d = ((ox - rx) ** 2 + (oy - ry) ** 2 + (oz - rz) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best = obj
        return best_d, best
