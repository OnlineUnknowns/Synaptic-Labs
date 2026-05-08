-- Phase 1 control and audit schema.

CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  ts_ms BIGINT NOT NULL,
  source_module TEXT NOT NULL,
  event_type TEXT NOT NULL,
  event_version TEXT NOT NULL,
  trace_id TEXT NOT NULL,
  priority TEXT NOT NULL,
  payload_json JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts_ms DESC);
CREATE INDEX IF NOT EXISTS idx_events_trace ON events (trace_id);
CREATE INDEX IF NOT EXISTS idx_events_type_ts ON events (event_type, ts_ms DESC);

CREATE TABLE IF NOT EXISTS decisions (
  decision_id TEXT PRIMARY KEY,
  ts_ms BIGINT NOT NULL,
  trace_id TEXT NOT NULL,
  selected_action TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
  veto_reason TEXT,
  context_hash TEXT NOT NULL,
  candidates_json JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions (ts_ms DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_trace ON decisions (trace_id);

CREATE TABLE IF NOT EXISTS rewards (
  reward_id TEXT PRIMARY KEY,
  ts_ms BIGINT NOT NULL,
  trace_id TEXT NOT NULL,
  source TEXT NOT NULL,
  value DOUBLE PRECISION NOT NULL,
  prediction_error DOUBLE PRECISION NOT NULL,
  components_json JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rewards_ts ON rewards (ts_ms DESC);
CREATE INDEX IF NOT EXISTS idx_rewards_trace ON rewards (trace_id);

CREATE TABLE IF NOT EXISTS safety_incidents (
  incident_id TEXT PRIMARY KEY,
  ts_ms BIGINT NOT NULL,
  severity TEXT NOT NULL,
  reason TEXT NOT NULL,
  action_blocked TEXT,
  resolution TEXT NOT NULL,
  trace_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_safety_ts ON safety_incidents (ts_ms DESC);
CREATE INDEX IF NOT EXISTS idx_safety_trace ON safety_incidents (trace_id);

CREATE TABLE IF NOT EXISTS module_versions (
  module_name TEXT PRIMARY KEY,
  artifact_version TEXT NOT NULL,
  artifact_digest TEXT NOT NULL,
  activated_at_ms BIGINT NOT NULL
);
