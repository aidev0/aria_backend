from __future__ import annotations

import json
from typing import Any

from .base_agent import AgentStatus, AgentType, BaseAgent, TaskResult


class PlannerAgent(BaseAgent):
    agent_type = AgentType.PLANNER
    name = "Planner"
    description = "Breaks down high-level requirements into structured development plans with actionable tasks."

    def get_system_prompt(self) -> str:
        return """You are Aria's Planner Agent — an expert software architect and project planner.

Your role:
- Take high-level requirements (from voice, text, or video context)
- Break them into structured, actionable development tasks
- Define clear acceptance criteria for each task
- Estimate complexity and dependencies between tasks
- Output a JSON development plan

Always respond with valid JSON in this format:
{
  "project_name": "string",
  "summary": "brief project summary",
  "tasks": [
    {
      "id": "task_1",
      "title": "string",
      "description": "detailed description",
      "type": "feature|bugfix|refactor|test|deploy|config",
      "priority": "high|medium|low",
      "dependencies": ["task_id"],
      "acceptance_criteria": ["criterion 1", "criterion 2"],
      "estimated_complexity": "simple|moderate|complex"
    }
  ],
  "tech_stack": ["technology 1", "technology 2"],
  "architecture_notes": "high-level architecture description"
}"""

    async def execute(self, task_input: dict[str, Any]) -> TaskResult:
        result = TaskResult(
            agent_type=self.agent_type,
            status=AgentStatus.WORKING,
            input_data=task_input,
        )

        requirement = task_input.get("requirement", "")
        context = task_input.get("context", "")
        model = task_input.get("model", "claude")

        prompt = f"""Create a detailed development plan for the following requirement:

REQUIREMENT:
{requirement}

{"ADDITIONAL CONTEXT:" + chr(10) + context if context else ""}

Analyze the requirement and produce a comprehensive development plan with clear, actionable tasks.
Respond ONLY with valid JSON."""

        response = await self.ask_ai(prompt, model=model)

        try:
            plan = json.loads(response)
        except json.JSONDecodeError:
            plan = {"raw_response": response}

        result.output_data = {"plan": plan}
        return result
