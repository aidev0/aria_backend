from __future__ import annotations

import json
from typing import Any

from .base_agent import AgentStatus, AgentType, BaseAgent, TaskResult


class TesterAgent(BaseAgent):
    agent_type = AgentType.TESTER
    name = "Tester"
    description = "Generates comprehensive test suites and validates code quality through automated testing."

    def get_system_prompt(self) -> str:
        return """You are Aria's Tester Agent — an expert QA engineer and test automation specialist.

Your role:
- Generate comprehensive test suites for code produced by the Developer agent
- Write unit tests, integration tests, and end-to-end tests
- Identify edge cases and potential failure modes
- Validate code meets acceptance criteria from the Planner
- Use appropriate testing frameworks (pytest, jest, XCTest, etc.)

Always respond with valid JSON in this format:
{
  "test_files": [
    {
      "path": "tests/test_file.ext",
      "content": "full test file content",
      "framework": "pytest|jest|xctest|etc",
      "test_count": 5
    }
  ],
  "test_commands": ["commands to run the tests"],
  "coverage_targets": ["list of functions/modules covered"],
  "edge_cases": ["edge case 1", "edge case 2"],
  "notes": "testing strategy notes"
}"""

    async def execute(self, task_input: dict[str, Any]) -> TaskResult:
        result = TaskResult(
            agent_type=self.agent_type,
            status=AgentStatus.WORKING,
            input_data=task_input,
        )

        code = task_input.get("code", {})
        task_spec = task_input.get("task_spec", {})
        acceptance_criteria = task_input.get("acceptance_criteria", [])
        model = task_input.get("model", "claude")

        prompt = f"""Generate a comprehensive test suite for the following code:

CODE:
{json.dumps(code, indent=2) if isinstance(code, dict) else code}

TASK SPECIFICATION:
{json.dumps(task_spec, indent=2) if isinstance(task_spec, dict) else task_spec}

ACCEPTANCE CRITERIA:
{json.dumps(acceptance_criteria) if acceptance_criteria else "None specified"}

Write thorough tests covering happy paths, edge cases, and error scenarios.
Respond ONLY with valid JSON."""

        response = await self.ask_ai(prompt, model=model)

        try:
            test_output = json.loads(response)
        except json.JSONDecodeError:
            test_output = {"raw_response": response}

        result.output_data = {"tests": test_output}
        return result
