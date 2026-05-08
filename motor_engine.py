"""
Motor engine: translates high-level actions into command envelopes, tracks
execution state, and reports outcomes for reward assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class MotorCommand:
    """Executable motor command with timing and safety metadata."""

    command_id: str
    action: str
    joint_targets: List[float]
    duration_ms: int
    speed_proxy: float
    stamp_ms: int


@dataclass
class MotorState:
    """Current motor execution snapshot."""

    last_command: Optional[MotorCommand]
    queue_depth: int
    history: List[MotorCommand] = field(default_factory=list)


class MotorEngine:
    """Discrete action execution model suitable for simulation or adapter bridge."""

    def __init__(self, *, ethics_speed_limit: float, horizon_ms: int = 250) -> None:
        self._ethics_limit = float(ethics_speed_limit)
        self._horizon_ms = int(horizon_ms)
        self._counter = 0
        self._last: Optional[MotorCommand] = None
        self._history: List[MotorCommand] = []

    @property
    def state(self) -> MotorState:
        return MotorState(last_command=self._last, queue_depth=0, history=list(self._history[-32:]))

    def plan_trajectory(self, action: str, *, stamp_ms: int) -> MotorCommand:
        """Create a command for `action` without executing."""
        dx, dy, dz = self._action_delta(action)
        step_len = max(1e-6, (dx**2 + dy**2 + dz**2) ** 0.5)
        speed_proxy = step_len / (self._horizon_ms / 1000.0)
        joint_targets = [dx, dy, dz, step_len]
        self._counter += 1
        return MotorCommand(
            command_id=f"cmd_{self._counter}",
            action=action,
            joint_targets=joint_targets,
            duration_ms=self._horizon_ms,
            speed_proxy=float(speed_proxy),
            stamp_ms=int(stamp_ms),
        )

    def execute_command(self, cmd: MotorCommand) -> MotorCommand:
        """Accept a planned command as executed (real systems would await feedback)."""
        if cmd.speed_proxy > self._ethics_limit + 1e-6:
            raise ValueError("motor_speed_exceeds_ethics_limit")
        self._last = cmd
        self._history.append(cmd)
        if len(self._history) > 256:
            self._history = self._history[-256:]
        return cmd

    @staticmethod
    def _action_delta(action: str) -> Tuple[float, float, float]:
        """Discrete action primitives in world units per command horizon."""
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
        return (0.5 * step, 0.1 * step, 0.0)

    def abort_command(self) -> None:
        """Stop current motion intent."""
        self._last = None

    def command_to_dict(self, cmd: MotorCommand) -> Dict[str, Any]:
        return {
            "commandId": cmd.command_id,
            "action": cmd.action,
            "jointTargets": cmd.joint_targets,
            "durationMs": cmd.duration_ms,
            "speedProxy": cmd.speed_proxy,
            "stampMs": cmd.stamp_ms,
        }

    def last_outcome_vector(self) -> Tuple[float, float, float, float]:
        """Return last motion delta vector for reward/progress estimation."""
        if not self._last:
            return (0.0, 0.0, 0.0, 0.0)
        j = self._last.joint_targets
        if len(j) < 4:
            j = j + [0.0] * (4 - len(j))
        return float(j[0]), float(j[1]), float(j[2]), float(j[3])
