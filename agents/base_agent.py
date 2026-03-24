from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    PLANNER = "planner"
    DEVELOPER = "developer"
    TESTER = "tester"
    CODE_REVIEWER = "code_reviewer"
    DEPLOYER = "deployer"
    REPORTER = "reporter"


class AgentStatus(str, Enum):
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskResult(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType
    status: AgentStatus = AgentStatus.IDLE
    input_data: dict[str, Any] = {}
    output_data: dict[str, Any] = {}
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class BaseAgent(ABC):
    """Base class for all Aria development agents."""

    agent_type: AgentType
    name: str
    description: str

    def __init__(self, ai_service):
        from services.ai_service import AIService

        self.ai_service: AIService = ai_service
        self.status = AgentStatus.IDLE
        self.current_task: TaskResult | None = None

    @abstractmethod
    async def execute(self, task_input: dict[str, Any]) -> TaskResult:
        """Execute the agent's primary task."""
        ...

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        ...

    async def run(self, task_input: dict[str, Any], model: str = "claude") -> TaskResult:
        """Run the agent on a task with status tracking."""
        self.current_task = TaskResult(
            agent_type=self.agent_type,
            status=AgentStatus.WORKING,
            input_data=task_input,
            started_at=datetime.now(timezone.utc),
        )
        self.status = AgentStatus.WORKING

        try:
            result = await self.execute(task_input)
            result.status = AgentStatus.COMPLETED
            result.completed_at = datetime.now(timezone.utc)
            self.status = AgentStatus.COMPLETED
            self.current_task = result
            return result
        except Exception as e:
            self.current_task.status = AgentStatus.FAILED
            self.current_task.error = str(e)
            self.current_task.completed_at = datetime.now(timezone.utc)
            self.status = AgentStatus.FAILED
            return self.current_task

    async def ask_ai(
        self,
        prompt: str,
        model: str = "claude",
        context: str | None = None,
    ) -> str:
        """Send a prompt to the AI service and get a response."""
        system_prompt = self.get_system_prompt()
        if context:
            system_prompt += f"\n\nAdditional context:\n{context}"
        return await self.ai_service.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
        )

    def get_status(self) -> dict[str, Any]:
        return {
            "agent_type": self.agent_type.value,
            "name": self.name,
            "status": self.status.value,
            "current_task": self.current_task.model_dump() if self.current_task else None,
        }
