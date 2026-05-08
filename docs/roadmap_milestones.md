# Roadmap Milestones (Sprint-Level)

## Release R1 - Architecture Skeleton (Sprints 1-2)

- Sprint 1:
  - repository scaffold and module boundary packages
  - shared contracts and event envelope baseline
  - orchestrator lifecycle state machine
- Sprint 2:
  - timing budget enforcement and watchdog hooks
  - telemetry skeleton and trace correlation ids

Verification gates:
- contract conformance tests pass
- lifecycle transition tests pass
- deadline budget violations are surfaced and audited

## Release R2 - Closed-Loop Simulation MVP (Sprints 3-4)

- Sprint 3:
  - world/self model ingest and focus window computation
  - decision and motor path connected with event bus
- Sprint 4:
  - scenario runner and baseline benchmark suite
  - deterministic replay harness

Verification gates:
- end-to-end loop executes for benchmark scenarios
- p95 decision-to-command latency meets target
- replay determinism delta within tolerance

## Release R3 - Memory, Reward, Synapse (Sprints 5-6)

- Sprint 5:
  - memory tier APIs and encode/recall baseline
  - reward computation with prediction error tracking
- Sprint 6:
  - synaptic update engine and bounded plasticity controls
  - learning telemetry and drift dashboard

Verification gates:
- learning signal quality metrics trend positively
- no unstable weight divergence in 8-hour soak
- memory recall precision exceeds baseline threshold

## Release R4 - Safety and Ethics Hardening (Sprints 7-8)

- Sprint 7:
  - ethics policy engine with formal invariant checks
  - emergency override protocol integration
- Sprint 8:
  - fault injection framework and safety incident exporter
  - operator-safe recovery workflow

Verification gates:
- all invariants S1-S6 pass scenario matrix
- emergency stop p99 latency under hard bound
- zero unauthorized motor command in chaos tests

## Release R5 - Predictive + Metacognition (Sprints 9-10)

- Sprint 9:
  - short-horizon predictive simulator integration
  - confidence estimation and calibration metrics
- Sprint 10:
  - strategy adjustment and degrade-mode triggers
  - uncertainty-aware decision weighting

Verification gates:
- calibration error below threshold
- degraded mode activates correctly under synthetic uncertainty spikes

## Release R6 - Sleep Consolidation and Continual Learning (Sprints 11-12)

- Sprint 11:
  - replay-based consolidation scheduler
  - memory pruning and synaptic normalization jobs
- Sprint 12:
  - anti-forgetting safeguards and rollback-capable model registry
  - counterfactual evaluation reports

Verification gates:
- catastrophic forgetting tests pass
- consolidation improves retention metrics without deadline regressions

## Release R7 - Hardware Integration and HIL (Sprints 13-14)

- Sprint 13:
  - hardware abstraction integration and real sensor adapters
  - actuator envelope enforcement on target hardware
- Sprint 14:
  - hardware-in-the-loop benchmarks and thermal/power profiling
  - failover drills and watchdog validation

Verification gates:
- HIL control stability targets met
- thermal and power budgets remain within envelope

## Release R8 - Production Readiness (Sprints 15-16)

- Sprint 15:
  - deployment automation, canary strategy, rollback runbooks
  - full observability and SLO alerting
- Sprint 16:
  - governance workflow for model/policy promotion
  - compliance and long-run operational validation

Verification gates:
- canary promotion policy validated end-to-end
- 24-hour stability soak with no critical regressions
- release checklist signed with safety and performance evidence
