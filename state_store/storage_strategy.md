# Storage Strategy and Versioning

## Technology Selection

- relational metadata and audit: PostgreSQL
- time-series telemetry: TimescaleDB (PostgreSQL extension)
- episodic trace documents: MongoDB
- semantic graph memory: Neo4j
- model artifacts and snapshots: S3-compatible object storage (MinIO in dev)
- event stream backbone: NATS JetStream (Phase 1), Kafka-compatible migration path (Phase 2+)

## Rationale

- PostgreSQL provides strong consistency for policy, config, and compliance audit.
- TimescaleDB keeps telemetry query patterns efficient and operationally simple.
- MongoDB supports heterogeneous episodic payloads and replay bundles.
- Neo4j fits semantic node-edge evolution and relationship traversal.
- Object storage decouples large binary model and snapshot lifecycle from transactional stores.

## Logical Data Domains

- `control_domain`: configs, module versions, safety policy revisions.
- `event_domain`: append-only event envelopes and replay indexes.
- `memory_domain`: episodic traces, semantic graph, procedural model pointers.
- `observability_domain`: metrics, traces, incident timelines.

## Migration and Versioning Policy

- relational migrations: Flyway, forward-only in production.
- document schema evolution: explicit `schema_version` in each document, lazy read-upgrade adapters.
- graph schema evolution: migration scripts with idempotent Cypher steps.
- event evolution:
  - immutable historical events
  - additive fields only for minor versions
  - breaking changes require new `eventType` or incremented `eventVersion` with transformer
- model versioning:
  - semantic versioning `major.minor.patch`
  - immutable artifact digests
  - canary validation before policy promotion

## Backup and Recovery

- snapshots:
  - PostgreSQL base backup every 6h, WAL continuous archiving.
  - MongoDB incremental snapshots every 2h.
  - Neo4j daily full + hourly transaction logs.
  - object storage versioning enabled.
- RPO target: 5 minutes.
- RTO target: 15 minutes for core control path.
- recovery flow:
  1. restore latest durable snapshot for each store
  2. replay event stream from checkpoint
  3. rebuild in-memory working memory cache
  4. run consistency check job before active mode

## Data Retention

- critical safety events: 7 years
- decision and reward traces: 180 days hot, 2 years cold
- high-rate telemetry: 14 days hot, downsampled 1 year
- episodic memory traces: salience-based retention with monthly pruning
