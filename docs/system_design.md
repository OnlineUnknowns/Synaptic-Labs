# System Design Specification

## Module Interaction Map

```mermaid
flowchart LR
  sensors[SensorsAdapters] --> worldModel[world_model_engine]
  sensors --> selfModel[self_model_engine]
  worldModel --> attention[attention_engine]
  selfModel --> attention
  attention --> predictive[predictive_engine]
  attention --> emotion[emotion_engine]
  predictive --> decision[decision_engine]
  emotion --> decision
  reward[reward_engine] --> decision
  ethics[ethics_engine] --> decision
  metacog[metacognition_engine] --> decision
  decision --> motor[motor_engine]
  motor --> actuators[ActuatorAdapters]
  decision --> memory[memory_engine]
  reward --> synapse[synapse_engine]
  memory --> synapse
  synapse --> nn[neural_network_layer]
  emergency[emergency_engine] -->|"veto_or_safe_override"| motor
  brainCore[brain_core] --> decision
```

## API Design (Internal Contracts)

- `PerceptionAPI`: `ingestSensorFrame`, `getWorldState`
- `AttentionAPI`: `computePrioritySet`, `getFocusContext`
- `DecisionAPI`: `proposeActions`, `evaluateActions`, `commitAction`
- `MotorAPI`: `planTrajectory`, `executeCommand`, `abortCommand`
- `MemoryAPI`: `storeTrace`, `recallByCue`, `consolidateBatch`
- `RewardAPI`: `computeReward`, `applyCreditAssignment`
- `SafetyAPI`: `evaluateConstraint`, `triggerEmergencyStop`

## Class Design (Conceptual)

- `BrainCoreOrchestrator`
- `ModuleContext`
- `NeuronGraphExecutor`
- `SynapticPlasticityManager`
- `MemoryManager`
- `DecisionArbiter`
- `EmotionStateController`
- `AttentionController`
- `SafetySupervisor`
- `MotorSupervisor`

## Event System Design

- Typed envelopes defined in `interfaces/event_schemas.json`
- channel priorities:
  - `critical`: safety and watchdog
  - `realtime`: perception-decision-motor
  - `background`: consolidation and analytics
- delivery:
  - exactly-once intent for critical path
  - at-least-once for background
- replay and dead-letter handling in state-store and bus adapters

## Cognitive State Machine

```mermaid
stateDiagram-v2
  [*] --> Boot
  Boot --> Calibrate
  Calibrate --> Idle
  Idle --> Perceive
  Perceive --> Attend
  Attend --> Predict
  Predict --> Decide
  Decide --> Execute
  Execute --> Evaluate
  Evaluate --> Perceive
  Evaluate --> SleepConsolidate: lowActivityWindow
  Decide --> SafeHold: ethicsOrEmergencyVeto
  Execute --> SafeHold: runtimeHazard
  SafeHold --> Recover
  Recover --> Idle
  Recover --> Shutdown: unrecoverableFault
  Shutdown --> [*]
```

## Execution Lifecycle

1. boot and load policy/model versions
2. calibrate sensors and establish self-state baseline
3. run continuous cognitive loop
4. update memory/synapse/reward learning signals per action outcome
5. preempt command flow on ethics or emergency veto
6. consolidate and prune memory during low-activity windows
7. persist snapshots and replayable events for fault recovery
