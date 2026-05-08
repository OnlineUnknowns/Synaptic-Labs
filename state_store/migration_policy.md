# Migration Policy

## Scope

This policy governs schema and contract changes across:

- relational schema (PostgreSQL/TimescaleDB)
- document schema (MongoDB episodic records)
- graph schema (Neo4j semantic memory)
- event schemas (`interfaces/event_schemas.json`)

## Rules

1. no destructive migration in-place on production hot path.
2. every migration is immutable and tracked with unique id.
3. backward compatibility window is required for one full release cycle.
4. event payload changes are additive unless a new event version is introduced.
5. migration scripts must include rollback notes even when rollback is operationally disallowed.

## Release Sequence

1. deploy readers that can handle old and new schema.
2. run migration in controlled window.
3. deploy writers for new schema.
4. enable strict validation mode.
5. remove legacy compatibility after deprecation period.

## Validation Gates

- migration dry-run in simulation dataset.
- checksum and row/document count validation.
- replay test for event-driven reconstruction.
- safety path latency regression must remain within hard deadlines.
