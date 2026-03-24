from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any


class Database:
    """MongoDB service using Motor (async driver). Works without MongoDB — falls back to in-memory."""

    def __init__(self):
        self._client = None
        self._db = None
        self._connected = False
        # In-memory fallback when MongoDB is not available
        self._mem_agent_configs: dict[str, dict] = {}
        self._mem_pipeline_runs: list[dict] = []
        self._mem_task_results: list[dict] = []

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self):
        uri = os.getenv("MONGODB_URI", "").strip()
        if not uri:
            print("[DB] No MONGODB_URI set — using in-memory storage")
            return

        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            self._client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
            # Test connection
            await self._client.admin.command("ping")
            self._db = self._client[os.getenv("MONGODB_DB", "aria")]
            self._connected = True
            print(f"[DB] Connected to MongoDB")
        except Exception as e:
            print(f"[DB] MongoDB connection failed: {e} — using in-memory storage")
            self._client = None
            self._db = None

    async def disconnect(self):
        if self._client:
            self._client.close()

    # ── Agent Configs ─────────────────────────────────────────────────────────

    async def get_agent_configs(self) -> list[dict]:
        if self._connected:
            cursor = self._db.agent_configs.find({}, {"_id": 0})
            return await cursor.to_list(length=100)
        return list(self._mem_agent_configs.values())

    async def get_agent_config(self, agent_type: str) -> dict | None:
        if self._connected:
            return await self._db.agent_configs.find_one(
                {"agent_type": agent_type}, {"_id": 0}
            )
        return self._mem_agent_configs.get(agent_type)

    async def set_agent_config(self, agent_type: str, config: dict) -> dict:
        config["agent_type"] = agent_type
        config["updated_at"] = datetime.now(timezone.utc).isoformat()

        if self._connected:
            await self._db.agent_configs.update_one(
                {"agent_type": agent_type},
                {"$set": config},
                upsert=True,
            )
        else:
            self._mem_agent_configs[agent_type] = config
        return config

    # ── Pipeline Runs ─────────────────────────────────────────────────────────

    async def save_pipeline_run(self, run_data: dict) -> str:
        run_data["created_at"] = datetime.now(timezone.utc).isoformat()
        if self._connected:
            result = await self._db.pipeline_runs.insert_one(run_data)
            return str(result.inserted_id)
        self._mem_pipeline_runs.append(run_data)
        return str(len(self._mem_pipeline_runs))

    async def get_pipeline_runs(self, limit: int = 20) -> list[dict]:
        if self._connected:
            cursor = (
                self._db.pipeline_runs
                .find({}, {"_id": 0})
                .sort("created_at", -1)
                .limit(limit)
            )
            return await cursor.to_list(length=limit)
        return self._mem_pipeline_runs[-limit:][::-1]

    # ── Task Results ──────────────────────────────────────────────────────────

    async def save_task_result(self, task_data: dict) -> str:
        task_data["created_at"] = datetime.now(timezone.utc).isoformat()
        if self._connected:
            result = await self._db.task_results.insert_one(task_data)
            return str(result.inserted_id)
        self._mem_task_results.append(task_data)
        return str(len(self._mem_task_results))

    async def get_task_results(
        self, agent_type: str | None = None, limit: int = 50
    ) -> list[dict]:
        if self._connected:
            query: dict[str, Any] = {}
            if agent_type:
                query["agent_type"] = agent_type
            cursor = (
                self._db.task_results
                .find(query, {"_id": 0})
                .sort("created_at", -1)
                .limit(limit)
            )
            return await cursor.to_list(length=limit)

        results = self._mem_task_results
        if agent_type:
            results = [r for r in results if r.get("agent_type") == agent_type]
        return results[-limit:][::-1]
