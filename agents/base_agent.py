from __future__ import annotations

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


class AgentConfig(BaseModel):
    """Per-agent configuration for model and CLI selection."""
    model: str = "claude"          # claude | gemini | openai
    cli: str = "none"              # claude | gemini | codex | none
    use_cli: bool = False          # Whether to use CLI instead of API


class TaskResult(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType
    status: AgentStatus = AgentStatus.IDLE
    input_data: dict[str, Any] = {}
    output_data: dict[str, Any] = {}
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    model_used: str | None = None
    cli_used: str | None = None


class BaseAgent(ABC):
    """Base class for all Aria development agents."""

    agent_type: AgentType
    name: str
    description: str

    def __init__(self, ai_service, cli_service=None):
        from services.ai_service import AIService
        from services.cli_service import CLIService

        self.ai_service: AIService = ai_service
        self.cli_service: CLIService | None = cli_service
        self.config = AgentConfig()
        self.status = AgentStatus.IDLE
        self.current_task: TaskResult | None = None

    def configure(self, config: dict) -> None:
        """Update agent configuration (model + CLI choice)."""
        self.config = AgentConfig(**{**self.config.model_dump(), **config})

    @abstractmethod
    async def execute(self, task_input: dict[str, Any]) -> TaskResult:
        """Execute the agent's primary task."""
        ...

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        ...

    async def run(self, task_input: dict[str, Any], model: str | None = None) -> TaskResult:
        """Run the agent on a task with status tracking."""
        # Use per-task model override, else agent config, else default
        effective_model = model or task_input.get("model") or self.config.model

        self.current_task = TaskResult(
            agent_type=self.agent_type,
            status=AgentStatus.WORKING,
            input_data=task_input,
            started_at=datetime.now(timezone.utc),
            model_used=effective_model,
            cli_used=self.config.cli if self.config.use_cli else None,
        )
        self.status = AgentStatus.WORKING

        try:
            # Override model in task_input so execute() sees it
            task_input["model"] = effective_model
            result = await self.execute(task_input)
            result.status = AgentStatus.COMPLETED
            result.completed_at = datetime.now(timezone.utc)
            result.model_used = effective_model
            result.cli_used = self.config.cli if self.config.use_cli else None
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
        model: str | None = None,
        context: str | None = None,
    ) -> str:
        """Send a prompt to the AI service (API) or CLI, depending on config."""
        effective_model = model or self.config.model
        system_prompt = self.get_system_prompt()
        if context:
            system_prompt += f"\n\nAdditional context:\n{context}"

        # If CLI mode is enabled and cli_service is available, use CLI
        if self.config.use_cli and self.cli_service and self.config.cli != "none":
            from services.cli_service import CLIType
            cli_type = CLIType(self.config.cli)
            full_prompt = f"{system_prompt}\n\n{prompt}"
            result = await self.cli_service.execute(cli_type, full_prompt)
            return result.get("stdout", result.get("stderr", "CLI error"))

        # Otherwise use API
        return await self.ai_service.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=effective_model,
        )

    def get_status(self) -> dict[str, Any]:
        return {
            "agent_type": self.agent_type.value,
            "name": self.name,
            "status": self.status.value,
            "config": self.config.model_dump(),
            "current_task": self.current_task.model_dump() if self.current_task else None,
        }
