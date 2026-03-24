from __future__ import annotations

import json
from typing import Any

from .base_agent import AgentStatus, AgentType, BaseAgent, TaskResult


class DeveloperAgent(BaseAgent):
    agent_type = AgentType.DEVELOPER
    name = "Developer"
    description = "Generates production-ready code based on task specifications from the Planner."

    def get_system_prompt(self) -> str:
        return """You are Aria's Developer Agent — an expert full-stack software engineer.

Your role:
- Take task specifications from the Planner agent
- Write clean, production-ready code
- Follow best practices and modern patterns
- Support multiple languages and frameworks
- Include inline comments for complex logic only

Always respond with valid JSON in this format:
{
  "files": [
    {
      "path": "relative/path/to/file.ext",
      "content": "full file content",
      "action": "create|modify|delete",
      "language": "python|typescript|swift|etc"
    }
  ],
  "commands": ["any shell commands needed (npm install, etc)"],
  "notes": "any important notes about the implementation"
}"""

    async def execute(self, task_input: dict[str, Any]) -> TaskResult:
        result = TaskResult(
            agent_type=self.agent_type,
            status=AgentStatus.WORKING,
            input_data=task_input,
        )

        task = task_input.get("task", {})
        plan_context = task_input.get("plan_context", "")
        existing_code = task_input.get("existing_code", "")
        model = task_input.get("model", "claude")

        prompt = f"""Implement the following task:

TASK:
{json.dumps(task, indent=2) if isinstance(task, dict) else task}

{"PROJECT CONTEXT:" + chr(10) + plan_context if plan_context else ""}
{"EXISTING CODE:" + chr(10) + existing_code if existing_code else ""}

Write clean, production-ready code. Respond ONLY with valid JSON."""

        response = await self.ask_ai(prompt, model=model)

        try:
            code_output = json.loads(response)
        except json.JSONDecodeError:
            code_output = {"raw_response": response}

        result.output_data = {"code": code_output}
        return result
