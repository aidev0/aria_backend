from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient


class Database:
    """MongoDB service using Motor (async driver)."""

    def __init__(self):
        self._client: AsyncIOMotorClient | None = None
        self._db = None

    async def connect(self):
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self._client = AsyncIOMotorClient(uri)
        self._db = self._client[os.getenv("MONGODB_DB", "aria")]

    async def disconnect(self):
        if self._client:
            self._client.close()

    @property
    def db(self):
        return self._db

    # ── Agent Configs ─────────────────────────────────────────────────────────

    async def get_agent_configs(self) -> list[dict]:
        """Get all agent configurations."""
        cursor = self.db.agent_configs.find({}, {"_id": 0})
        return await cursor.to_list(length=100)

    async def get_agent_config(self, agent_type: str) -> dict | None:
        """Get config for a specific agent."""
        return await self.db.agent_configs.find_one(
            {"agent_type": agent_type}, {"_id": 0}
        )

    async def set_agent_config(self, agent_type: str, config: dict) -> dict:
        """Set/update config for an agent (model + CLI choice)."""
        config["agent_type"] = agent_type
        config["updated_at"] = datetime.now(timezone.utc).isoformat()

        await self.db.agent_configs.update_one(
            {"agent_type": agent_type},
            {"$set": config},
            upsert=True,
        )
        return config

    # ── Pipeline Runs ─────────────────────────────────────────────────────────

    async def save_pipeline_run(self, run_data: dict) -> str:
        """Save a pipeline run result."""
        run_data["created_at"] = datetime.now(timezone.utc).isoformat()
        result = await self.db.pipeline_runs.insert_one(run_data)
        return str(result.inserted_id)

    async def get_pipeline_runs(self, limit: int = 20) -> list[dict]:
        """Get recent pipeline runs."""
        cursor = (
            self.db.pipeline_runs
            .find({}, {"_id": 0})
            .sort("created_at", -1)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    # ── Task Results ──────────────────────────────────────────────────────────

    async def save_task_result(self, task_data: dict) -> str:
        """Save an individual task result."""
        task_data["created_at"] = datetime.now(timezone.utc).isoformat()
        result = await self.db.task_results.insert_one(task_data)
        return str(result.inserted_id)

    async def get_task_results(
        self, agent_type: str | None = None, limit: int = 50
    ) -> list[dict]:
        """Get recent task results, optionally filtered by agent."""
        query: dict[str, Any] = {}
        if agent_type:
            query["agent_type"] = agent_type

        cursor = (
            self.db.task_results
            .find(query, {"_id": 0})
            .sort("created_at", -1)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)
