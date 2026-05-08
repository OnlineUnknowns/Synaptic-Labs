# Safety and Ethics Architecture

## Decision Authority Order

1. `emergency_engine` hard override (highest authority).
2. `ethics_engine` hard veto on forbidden actions.
3. `decision_engine` utility arbitration within allowed constraints.
4. `motor_engine` execution only for approved command contracts.

## Veto Policies

- `emergency_engine` triggers immediate `hold`, `stop`, or `shutdown`.
- `ethics_engine` enforces:
  - forbidden zones or entities
  - force/velocity limits per context
  - mission policy constraints
  - operator override policy validity
- if either engine returns veto, action cannot proceed.

## Formal Invariants (Phase 1)

- **Invariant-S1**: no motor command is issued without a successful ethics check in the same or newer tick context.
- **Invariant-S2**: emergency critical state preempts all queued realtime/background events.
- **Invariant-S3**: actuator command envelope always respects kinematic and thermal limits.
- **Invariant-S4**: unresolved sensor-confidence below threshold forces degrade mode or hold.
- **Invariant-S5**: inability to evaluate safety constraints defaults to deny.
- **Invariant-S6**: every veto/override emits an immutable audit event with reason and trace id.

## Runtime Safety States

- `nominal`: standard operation.
- `guarded`: increased caution after moderate risk trends.
- `safe_hold`: zero-motion hold with active sensing and operator notification.
- `emergency_stop`: immediate controlled halt.
- `shutdown`: full actuator disable except watchdog and telemetry.

## Verification Requirements

- property-based tests for invariants S1-S6.
- scenario fault injection tests (sensor blackout, actuator stall, map inconsistency).
- latency verification: emergency detection to stop command < 20 ms end-to-end.
- ethics policy regression tests with deterministic fixtures.
