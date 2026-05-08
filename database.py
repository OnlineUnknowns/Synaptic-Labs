"""
Persistence layer: SQLAlchemy models, engine/session factory, and repository helpers.

Supports SQLite (default) and PostgreSQL via `DATABASE_URL`. Uses synchronous
sessions for simplicity and broad compatibility with ASGI worker patterns.
"""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Iterator, Optional

from sqlalchemy import BigInteger, Float, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from config import get_settings


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class EventRow(Base):
    """Append-only event log aligned with the architecture event envelope."""

    __tablename__ = "events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ts_ms: Mapped[int] = mapped_column(BigInteger, index=True)
    source_module: Mapped[str] = mapped_column(String(128), index=True)
    event_type: Mapped[str] = mapped_column(String(128), index=True)
    event_version: Mapped[str] = mapped_column(String(16))
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    priority: Mapped[str] = mapped_column(String(32))
    payload_json: Mapped[str] = mapped_column(Text)


class DecisionRow(Base):
    """Committed decisions for audit and replay."""

    __tablename__ = "decisions"

    decision_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ts_ms: Mapped[int] = mapped_column(BigInteger, index=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    selected_action: Mapped[str] = mapped_column(String(256))
    confidence: Mapped[float] = mapped_column(Float)
    veto_reason: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    context_hash: Mapped[str] = mapped_column(String(128))
    candidates_json: Mapped[str] = mapped_column(Text)


class RewardRow(Base):
    """Reward signals and components."""

    __tablename__ = "rewards"

    reward_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ts_ms: Mapped[int] = mapped_column(BigInteger, index=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(128))
    value: Mapped[float] = mapped_column(Float)
    prediction_error: Mapped[float] = mapped_column(Float)
    components_json: Mapped[str] = mapped_column(Text)


class SafetyIncidentRow(Base):
    """Safety and ethics incidents."""

    __tablename__ = "safety_incidents"

    incident_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ts_ms: Mapped[int] = mapped_column(BigInteger, index=True)
    severity: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(Text)
    action_blocked: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    resolution: Mapped[str] = mapped_column(String(64))
    trace_id: Mapped[str] = mapped_column(String(64), index=True)


class ModuleVersionRow(Base):
    """Registered module versions for reproducibility."""

    __tablename__ = "module_versions"

    module_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    artifact_version: Mapped[str] = mapped_column(String(64))
    artifact_digest: Mapped[str] = mapped_column(String(128))
    activated_at_ms: Mapped[int] = mapped_column(BigInteger)


_engine = None
_SessionLocal: Optional[sessionmaker[Session]] = None


def _utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def get_engine():
    """Lazily construct the SQLAlchemy engine (singleton)."""
    global _engine
    if _engine is None:
        settings = get_settings()
        connect_args = {}
        if settings.database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return a sessionmaker bound to the shared engine."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    return _SessionLocal


def init_db() -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(bind=get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope around a series of operations."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def new_id(prefix: str = "") -> str:
    """Generate a unique identifier suitable for primary keys."""
    return f"{prefix}{uuid.uuid4().hex}"


def persist_event(
    *,
    source_module: str,
    event_type: str,
    event_version: str,
    trace_id: str,
    priority: str,
    payload: Dict[str, Any],
) -> str:
    """Insert an event row. Returns event_id."""
    event_id = new_id("evt_")
    row = EventRow(
        event_id=event_id,
        ts_ms=_utc_now_ms(),
        source_module=source_module,
        event_type=event_type,
        event_version=event_version,
        trace_id=trace_id,
        priority=priority,
        payload_json=json.dumps(payload, separators=(",", ":"), default=str),
    )
    with session_scope() as s:
        s.add(row)
    return event_id


def persist_decision(
    *,
    decision_id: str,
    trace_id: str,
    selected_action: str,
    confidence: float,
    veto_reason: Optional[str],
    context_hash: str,
    candidates: Dict[str, Any],
) -> None:
    with session_scope() as s:
        s.add(
            DecisionRow(
                decision_id=decision_id,
                ts_ms=_utc_now_ms(),
                trace_id=trace_id,
                selected_action=selected_action,
                confidence=confidence,
                veto_reason=veto_reason,
                context_hash=context_hash,
                candidates_json=json.dumps(candidates, separators=(",", ":"), default=str),
            )
        )


def persist_reward(
    *,
    reward_id: str,
    trace_id: str,
    source: str,
    value: float,
    prediction_error: float,
    components: Dict[str, Any],
) -> None:
    with session_scope() as s:
        s.add(
            RewardRow(
                reward_id=reward_id,
                ts_ms=_utc_now_ms(),
                trace_id=trace_id,
                source=source,
                value=value,
                prediction_error=prediction_error,
                components_json=json.dumps(components, separators=(",", ":"), default=str),
            )
        )


def persist_safety_incident(
    *,
    incident_id: str,
    severity: str,
    reason: str,
    action_blocked: Optional[str],
    resolution: str,
    trace_id: str,
) -> None:
    with session_scope() as s:
        s.add(
            SafetyIncidentRow(
                incident_id=incident_id,
                ts_ms=_utc_now_ms(),
                severity=severity,
                reason=reason,
                action_blocked=action_blocked,
                resolution=resolution,
                trace_id=trace_id,
            )
        )


def upsert_module_version(*, module_name: str, artifact_version: str, artifact_digest: str) -> None:
    with session_scope() as s:
        row = s.get(ModuleVersionRow, module_name)
        if row is None:
            row = ModuleVersionRow(
                module_name=module_name,
                artifact_version=artifact_version,
                artifact_digest=artifact_digest,
                activated_at_ms=_utc_now_ms(),
            )
            s.add(row)
        else:
            row.artifact_version = artifact_version
            row.artifact_digest = artifact_digest
            row.activated_at_ms = _utc_now_ms()


class Database:
    """
    Thin facade used by the API layer for initialization and health checks.

    Keeps persistence concerns discoverable without importing global functions
    throughout the codebase.
    """

    def __init__(self) -> None:
        self._initialized = False

    def initialize(self) -> None:
        if not self._initialized:
            init_db()
            self._initialized = True

    def health(self) -> Dict[str, Any]:
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True, "database_url_scheme": engine.url.drivername}
