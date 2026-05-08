# Phase 1 MVP Specification

## Scope

Closed-loop simulation MVP with these active modules:

- `brain_core`
- `world_model_engine`
- `self_model_engine`
- `attention_engine`
- `predictive_engine`
- `emotion_engine`
- `decision_engine`
- `ethics_engine`
- `emergency_engine`
- `motor_engine`
- `reward_engine`
- `memory_engine`
- `synapse_engine`
- `metacognition_engine`

## Cognitive Loop (MVP)

1. ingest simulated sensor frame
2. update world and self models
3. compute attention focus window
4. predict outcomes for top action candidates
5. evaluate ethics and emergency constraints
6. commit decision and issue motor command
7. observe outcome and compute reward
8. update memory traces and synaptic weights
9. log metrics and audit events

## Benchmark Scenarios

- obstacle navigation with moving hazards
- object retrieval with partial occlusion
- low-battery return-to-base
- policy conflict requiring ethics veto
- induced sensor uncertainty and fallback recovery

## Acceptance Criteria

### Functional

- all listed modules exchange events through typed envelopes.
- decision path enforces ethics and emergency checks before motor command.
- memory encode and reward update occur on each completed action.
- emergency override can interrupt active motor execution.

### Performance

- safety reaction time: p99 < 20 ms (hazard detect to stop command).
- motor command issuance latency: p99 < 10 ms after decision commit.
- control loop stability: no deadline miss in 99.5% of ticks over 30-minute run.
- event loss on critical channel: 0 tolerated.

### Learning Quality

- reward prediction error trend improves over repeated trials.
- collision rate decreases by at least 30% between first and last quartile of run.
- successful task completion improves by at least 20% with memory enabled vs disabled baseline.

### Reliability

- recovery from simulated process restart via snapshot + replay in < 60 seconds.
- no invariant violations S1-S6 across benchmark suite.

## Deliverables

- reproducible simulation profile and seed list
- benchmark run report with latency and success metrics
- safety incident audit export
- model and memory state snapshots for replay
