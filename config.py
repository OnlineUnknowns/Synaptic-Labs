"""
Application configuration for the Brain-Inspired Robotics Cognitive System.

Uses pydantic-settings for environment-driven configuration with safe defaults
suitable for local development (SQLite). Production deployments should set
DATABASE_URL to PostgreSQL and configure CORS origins explicitly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings loaded from environment variables and optional `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="Brain Cognitive System")
    environment: str = Field(default="development", description="development|staging|production")
    log_level: str = Field(default="INFO")

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_cors_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins, or * for dev",
    )

    database_url: str = Field(
        default="sqlite:///./brain_cognitive.db",
        description="SQLAlchemy database URL (SQLite or PostgreSQL)",
    )

    cognitive_tick_budget_ms: float = Field(
        default=50.0,
        ge=1.0,
        description="Soft budget for a full cognitive cycle (logging/metrics).",
    )
    motor_command_timeout_ms: float = Field(default=5000.0, ge=1.0)

    emergency_distance_threshold: float = Field(
        default=0.35,
        ge=0.0,
        description="World-units distance below which proximity is treated as critical hazard.",
    )
    ethics_max_speed: float = Field(default=1.0, ge=0.0)

    synapse_learning_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    synapse_eligibility_decay: float = Field(default=0.9, ge=0.0, lt=1.0)
    synapse_weight_clip: float = Field(default=2.0, gt=0.0)

    memory_working_capacity: int = Field(default=64, ge=8)
    memory_episodic_max: int = Field(default=10_000, ge=100)

    persist_events: bool = Field(
        default=True,
        description="If True, append key lifecycle events to the database.",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        raw = self.api_cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    @field_validator("database_url")
    @classmethod
    def _validate_db_url(cls, v: str) -> str:
        if not v or "://" not in v:
            raise ValueError("database_url must be a valid SQLAlchemy URL")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton for import-time performance."""
    return Settings()
