"""
Brain core: orchestrates the cognitive architecture for one robot instance.

Executes a full perceive → attend → predict → decide → execute → learn cycle with
deterministic safety preemption. This module is the integration point used by the
HTTP API and `main.py`.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from attention_engine import AttentionEngine
from config import Settings, get_settings
from decision_engine import DecisionEngine, DecisionResult
from emergency_engine import EmergencyEngine, EmergencyLevel, EmergencyStatus
from emotion_engine import EmotionEngine, EmotionModulation
from ethics_engine import EthicsEngine
from memory_engine import MemoryEngine
from metacognition_engine import MetacognitionEngine
from motor_engine import MotorEngine, MotorCommand
from personality_engine import PersonalityEngine
from predictive_engine import PredictiveEngine
from reward_engine import RewardEngine
from self_model_engine import SelfModelEngine
from sleep_consolidation_engine import SleepConsolidationEngine
from synapse_engine import SynapseEngine
from world_model_engine import WorldModelEngine


class LifecycleState(str, Enum):
    BOOT = "boot"
    CALIBRATE = "calibrate"
    IDLE = "idle"
    ACTIVE = "active"
    SAFE_HOLD = "safe_hold"
    SLEEP_CONSOLIDATE = "sleep_consolidate"


@dataclass
class CycleInput:
    """Structured input for one cognitive cycle."""

    sensor_frame: Dict[str, Any]
    telemetry: Dict[str, Any]
    goal_keywords: Sequence[str]
    goal_position: Tuple[float, float, float]
    robot_position: Tuple[float, float, float]
    stamp_ms: int


class BrainCore:
    """
    Single-process cognitive runtime (Phase 1).

    Thread-safety: not synchronized; use one instance per robot or protect externally.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

        self.world = WorldModelEngine()
        self.self_model = SelfModelEngine()
        self.personality = PersonalityEngine()
        self.emotion = EmotionEngine()
        self.attention = AttentionEngine(capacity=max(4, self.settings.memory_working_capacity // 8))
        self.predictive = PredictiveEngine(horizon_ms=250)
        self.meta = MetacognitionEngine(window=32)
        self.ethics = EthicsEngine(max_speed=self.settings.ethics_max_speed)
        self.emergency = EmergencyEngine(distance_threshold=self.settings.emergency_distance_threshold)
        self.memory = MemoryEngine(
            working_capacity=self.settings.memory_working_capacity,
            episodic_max=self.settings.memory_episodic_max,
        )
        self.synapse = SynapseEngine(
            learning_rate=self.settings.synapse_learning_rate,
            eligibility_decay=self.settings.synapse_eligibility_decay,
            weight_clip=self.settings.synapse_weight_clip,
        )
        self.reward = RewardEngine()
        self.motor = MotorEngine(ethics_speed_limit=self.ethics.speed_limit, horizon_ms=250)
        self.decision = DecisionEngine(self.ethics)
        self.sleep = SleepConsolidationEngine()

        self._tick = 0
        self._lifecycle = LifecycleState.BOOT
        self._low_activity_streak = 0
        self._last_task_progress = 0.0
        self._last_hazard_proximity = 0.0

    @property
    def lifecycle(self) -> LifecycleState:
        return self._lifecycle

    def initialize(self) -> None:
        """Transition from BOOT to IDLE (calibration stub for Phase 1)."""
        self._lifecycle = LifecycleState.CALIBRATE
        # Phase 1: calibration is a no-op beyond self/world reset readiness.
        self._lifecycle = LifecycleState.IDLE

    def snapshot(self) -> Dict[str, Any]:
        """Lightweight runtime snapshot for observability endpoints."""
        ws = self.world.state
        return {
            "tick": self._tick,
            "lifecycle": self._lifecycle.value,
            "worldStateId": getattr(ws, "world_state_id", None),
            "worldUncertainty": getattr(ws, "uncertainty", None),
            "objectCount": len(getattr(ws, "objects", []) or []),
            "selfConfidence": self.self_model.state.confidence,
            "battery": self.self_model.state.battery,
            "emotion": self.emotion.state.to_dict(),
            "valueEstimate": self.reward.value_estimate,
            "workingMemory": len(self.memory.working),
            "episodicMemory": len(self.memory.episodic),
            "motorLastCommand": self.motor.state.last_command.action if self.motor.state.last_command else None,
        }

    def run_cycle(self, inp: CycleInput) -> Dict[str, Any]:
        """Execute one full cognitive cycle and return an audit-friendly dict."""
        self._tick += 1
        trace_id = uuid.uuid4().hex
        self._lifecycle = LifecycleState.ACTIVE

        world_state = self.world.ingest_sensor_frame(inp.sensor_frame)
        self_state = self.self_model.ingest_telemetry(inp.telemetry)

        emergency_status = self.emergency.evaluate(
            world=self.world,
            self_state=self_state,
            robot_position=inp.robot_position,
        )
        if emergency_status.level != EmergencyLevel.NONE:
            self.emotion.apply_hazard(1.0 if emergency_status.level == EmergencyLevel.CRITICAL else 0.55)

        self.emotion.step_decay()

        novelty_hints: List[float] = []
        for obj in world_state.objects[:8]:
            novelty_hints.append(self.memory.novelty_hint_for_object(obj.object_id, obj.label))
        memory_novelty_hint = float(sum(novelty_hints) / max(1, len(novelty_hints)))

        focus = self.attention.compute_focus(
            world=world_state,
            goal_keywords=inp.goal_keywords,
            personality=self.personality.profile,
            memory_novelty_hint=memory_novelty_hint,
        )

        meta_assessment = self.meta.assess(
            world_uncertainty=world_state.uncertainty,
            self_confidence=self_state.confidence,
        )

        modulation: EmotionModulation = self.emotion.compute_modulation(self.personality.profile.risk_tolerance)

        candidates = self.decision.propose_actions(
            recommend_more_sensing=meta_assessment.recommend_more_sensing,
            exploration=float(modulation.exploration_multiplier),
        )

        pred_batch = self.predictive.predict_batch(
            world=world_state,
            self_state=self_state,
            robot_position=inp.robot_position,
            goal_position=inp.goal_position,
            candidates=candidates,
            emotion=modulation,
            ethics_speed_limit=self.ethics.speed_limit,
        )

        focus_labels = [it.label for it in focus.items]
        proposed_speed = float(max((p.notes.get("speed_proxy", 0.0) for p in pred_batch.predictions), default=0.0))
        ethics_result = self.ethics.evaluate_candidates(
            candidates=[p.action for p in pred_batch.predictions],
            focus_labels=focus_labels,
            proposed_speed=proposed_speed,
        )

        scores, ordered_actions = self.decision.evaluate_actions(
            batch=pred_batch,
            emotion=modulation,
            meta=meta_assessment,
            personality=self.personality.profile,
            ethics_result=ethics_result,
        )

        decision = self.decision.commit_action(
            actions=ordered_actions,
            scores=scores,
            ethics_result=ethics_result,
        )

        final_action = decision.selected_action
        if self.emergency.should_veto_motion(emergency_status):
            final_action = self._map_safe_action(emergency_status)
            decision = DecisionResult(
                decision_id=decision.decision_id,
                selected_action=final_action,
                confidence=min(decision.confidence, 0.3),
                candidates=decision.candidates,
                scores=decision.scores,
                ethics=decision.ethics,
                veto_reason=f"emergency:{emergency_status.reason}",
            )

        cmd = self.motor.plan_trajectory(final_action, stamp_ms=inp.stamp_ms)
        try:
            self.motor.execute_command(cmd)
        except ValueError:
            # Ethics speed violation at motor layer: force stay.
            cmd = self.motor.plan_trajectory("stay", stamp_ms=inp.stamp_ms)
            self.motor.execute_command(cmd)
            final_action = "stay"

        self.self_model.register_motor_stress(cmd.speed_proxy)

        task_progress, hazard_proximity = self._estimate_progress_and_hazard(
            inp.robot_position,
            inp.goal_position,
            cmd,
            emergency_status,
        )
        self._last_task_progress = task_progress
        self._last_hazard_proximity = hazard_proximity

        reward_total, breakdown, pred_err = self.reward.compute_reward(
            task_progress=task_progress,
            hazard_proximity=hazard_proximity,
            battery=self_state.battery,
            novelty=memory_novelty_hint,
            ethics_violation=not ethics_result.allowed,
            emergency_level=emergency_status.level.value,
        )
        self.meta.observe_prediction_error(pred_err)
        self.emotion.apply_reward(reward_total)

        focus_token = focus.items[0].item_id if focus.items else "none"
        self.reward.apply_credit_assignment(
            prediction_error=pred_err,
            action_name=final_action,
            focus_token=focus_token,
            synapse_on_coactivation=self.synapse.on_coactivation,
        )
        plasticity = self.synapse.apply_reward_error(pred_err)

        salience = min(1.0, 0.25 + abs(reward_total) + 0.35 * hazard_proximity)
        self.memory.store_trace(
            content=f"action={final_action};reward={reward_total:.3f}",
            tags=[final_action, focus_token, *inp.goal_keywords],
            salience=salience,
            emotion_tag=self.emotion.state.valence >= 0.5 and "positive" or "negative",
        )

        self._update_low_activity_and_maybe_consolidate(final_action)

        ctx_hash = self.memory.context_hash(
            [trace_id, world_state.world_state_id, final_action, str(round(reward_total, 3))]
        )

        out: Dict[str, Any] = {
            "traceId": trace_id,
            "tick": self._tick,
            "lifecycle": self._lifecycle.value,
            "world": {
                "worldStateId": world_state.world_state_id,
                "uncertainty": world_state.uncertainty,
                "objects": len(world_state.objects),
            },
            "self": {
                "battery": self_state.battery,
                "confidence": self_state.confidence,
                "thermal": self_state.thermal,
            },
            "attention": self.attention.focus_to_dict(focus),
            "prediction": self.predictive.batch_to_dict(pred_batch),
            "metacognition": self.meta.to_dict(meta_assessment),
            "emotion": self.emotion.to_audit_dict() | {"modulation": asdict(modulation)},
            "ethics": self.ethics.to_dict(ethics_result),
            "emergency": self.emergency.to_dict(emergency_status),
            "decision": self.decision.result_to_dict(decision),
            "motor": self.motor.command_to_dict(cmd),
            "reward": {
                "total": reward_total,
                "breakdown": breakdown.__dict__,
                "predictionError": pred_err,
            },
            "learning": plasticity.__dict__,
            "consolidation": getattr(self, "_last_consolidation", None),
        }

        if self.settings.persist_events:
            self._persist_optional(
                trace_id=trace_id,
                decision=decision,
                reward_total=reward_total,
                pred_err=pred_err,
                breakdown=breakdown,
                emergency_status=emergency_status,
                ctx_hash=ctx_hash,
            )

        return out

    def _map_safe_action(self, status: EmergencyStatus) -> str:
        if status.safe_action in ("stop", "shutdown"):
            return "stay"
        if status.safe_action == "hold":
            return "stay"
        return "stay"

    def _estimate_progress_and_hazard(
        self,
        robot_pos: Tuple[float, float, float],
        goal_pos: Tuple[float, float, float],
        cmd: MotorCommand,
        emergency_status: EmergencyStatus,
    ) -> Tuple[float, float]:
        rx, ry, rz = robot_pos
        gx, gy, gz = goal_pos
        dx, dy, dz, _step = self.motor.last_outcome_vector()
        nx, ny, nz = rx + dx, ry + dy, rz + dz

        dist_before = max(1e-6, ((gx - rx) ** 2 + (gy - ry) ** 2 + (gz - rz) ** 2) ** 0.5)
        dist_after = max(1e-6, ((gx - nx) ** 2 + (gy - ny) ** 2 + (gz - nz) ** 2) ** 0.5)
        task_progress = float(dist_before - dist_after)

        d, obj = self.world.nearest_hazard((nx, ny, nz))
        if obj is None:
            hazard_prox = 0.0
        else:
            hazard_prox = float(max(0.0, min(1.0, 1.0 - (d / max(1e-6, self.settings.emergency_distance_threshold * 2.0)))))
            hazard_prox = max(hazard_prox, obj.risk * 0.65)

        if emergency_status.level == EmergencyLevel.CRITICAL:
            hazard_prox = max(hazard_prox, 0.95)
        elif emergency_status.level == EmergencyLevel.HIGH:
            hazard_prox = max(hazard_prox, 0.75)

        return task_progress, hazard_prox

    def _update_low_activity_and_maybe_consolidate(self, final_action: str) -> None:
        if final_action.lower() in ("stay", "hold", "stop", "s"):
            self._low_activity_streak += 1
        else:
            self._low_activity_streak = 0

        if self._low_activity_streak >= 12:
            self._lifecycle = LifecycleState.SLEEP_CONSOLIDATE
            report = self.sleep.run(memory=self.memory, synapse=self.synapse)
            self._last_consolidation = self.sleep.report_to_dict(report)
            self._low_activity_streak = 0
            self._lifecycle = LifecycleState.IDLE
        else:
            self._last_consolidation = None

    def _persist_optional(
        self,
        *,
        trace_id: str,
        decision: DecisionResult,
        reward_total: float,
        pred_err: float,
        breakdown: Any,
        emergency_status: EmergencyStatus,
        ctx_hash: str,
    ) -> None:
        # Import locally to allow running without DB drivers in constrained envs.
        from database import persist_decision, persist_event, persist_reward, persist_safety_incident

        persist_event(
            source_module="brain_core",
            event_type="CycleCompleted",
            event_version="v1",
            trace_id=trace_id,
            priority="realtime",
            payload={"tick": self._tick, "decisionId": decision.decision_id},
        )
        persist_decision(
            decision_id=decision.decision_id,
            trace_id=trace_id,
            selected_action=decision.selected_action,
            confidence=float(decision.confidence),
            veto_reason=decision.veto_reason,
            context_hash=ctx_hash,
            candidates={"candidates": decision.candidates, "scores": decision.scores},
        )
        persist_reward(
            reward_id=f"rew_{decision.decision_id}",
            trace_id=trace_id,
            source="reward_engine",
            value=float(reward_total),
            prediction_error=float(pred_err),
            components=breakdown.__dict__,
        )
        if emergency_status.level in (EmergencyLevel.HIGH, EmergencyLevel.CRITICAL):
            persist_safety_incident(
                incident_id=f"inc_{trace_id}_{self._tick}",
                severity=emergency_status.level.value,
                reason=emergency_status.reason,
                action_blocked=decision.selected_action,
                resolution=emergency_status.safe_action,
                trace_id=trace_id,
            )
