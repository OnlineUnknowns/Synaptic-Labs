"""
HTTP API (FastAPI) exposing the cognitive runtime for integration and testing.

Endpoints support one-shot cycle execution (simulation-friendly), runtime
snapshot retrieval, and health checks including database connectivity.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from brain_core import BrainCore, CycleInput
from config import get_settings
from database import Database


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db = Database()
        db.initialize()
        brain = BrainCore(settings=settings)
        brain.initialize()
        app.state.db = db
        app.state.brain = brain
        yield

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Brain-inspired robotics cognitive architecture - Phase 1 runtime API",
        lifespan=lifespan,
    )

    origins = settings.cors_origins_list
    allow_credentials = False if origins == ["*"] else True
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _brain(request: Request) -> BrainCore:
        return request.app.state.brain

    def _db(request: Request) -> Database:
        return request.app.state.db

    class GoalPosition(BaseModel):
        x: float = 1.0
        y: float = 0.0
        z: float = 0.0

    class RobotPosition(BaseModel):
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0

    class CycleRequest(BaseModel):
        """Request body for executing one cognitive cycle."""

        sensor_frame: Dict[str, Any] = Field(default_factory=dict)
        telemetry: Dict[str, Any] = Field(default_factory=dict)
        goal_keywords: List[str] = Field(default_factory=lambda: ["goal"])
        goal: GoalPosition = Field(default_factory=GoalPosition)
        robot: RobotPosition = Field(default_factory=RobotPosition)
        stamp_ms: Optional[int] = Field(
            default=None,
            description="Unix epoch milliseconds; defaults to server time if omitted.",
        )

        model_config = {"extra": "allow"}

    @app.get("/health")
    def health(request: Request) -> Dict[str, Any]:
        """Liveness and dependency checks."""
        db_health = _db(request).health()
        return {"status": "ok", "environment": settings.environment, "database": db_health}

    @app.get("/v1/snapshot")
    def snapshot(request: Request) -> Dict[str, Any]:
        """Return a compact cognitive runtime snapshot."""
        return _brain(request).snapshot()

    @app.post("/v1/cycle")
    def run_cycle(request: Request, body: CycleRequest) -> Dict[str, Any]:
        """
        Execute one full cognitive cycle.

        Example `sensor_frame`:
        ```json
        {
          "timestamp_ms": 1715212800000,
          "localization_confidence": 0.92,
          "objects": [
            {"id": "o1", "label": "obstacle", "position": [0.6, 0.0, 0.0], "risk": 0.8, "saliency": 0.7}
          ]
        }
        ```
        """
        stamp = body.stamp_ms if body.stamp_ms is not None else int(time.time() * 1000)
        gp: Tuple[float, float, float] = (body.goal.x, body.goal.y, body.goal.z)
        rp: Tuple[float, float, float] = (body.robot.x, body.robot.y, body.robot.z)

        inp = CycleInput(
            sensor_frame=body.sensor_frame,
            telemetry=body.telemetry,
            goal_keywords=body.goal_keywords,
            goal_position=gp,
            robot_position=rp,
            stamp_ms=stamp,
        )
        return _brain(request).run_cycle(inp)

    @app.post("/v1/consolidate")
    def consolidate(request: Request) -> Dict[str, Any]:
        """Manually trigger sleep consolidation (useful for testing)."""
        brain = _brain(request)
        report = brain.sleep.run(memory=brain.memory, synapse=brain.synapse)
        return brain.sleep.report_to_dict(report)

    return app


app = create_app()
