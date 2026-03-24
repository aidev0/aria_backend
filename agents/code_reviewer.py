from __future__ import annotations

import json
from typing import Any

from .base_agent import AgentStatus, AgentType, BaseAgent, TaskResult


class CodeReviewerAgent(BaseAgent):
    agent_type = AgentType.CODE_REVIEWER
    name = "Code Reviewer"
    description = "Reviews code for quality, security, performance, and best practices."

    def get_system_prompt(self) -> str:
        return """You are Aria's Code Reviewer Agent — a senior engineer specializing in code review.

Your role:
- Review code for quality, readability, and maintainability
- Identify security vulnerabilities (OWASP top 10, injection, XSS, etc.)
- Check for performance issues and optimization opportunities
- Verify adherence to best practices and design patterns
- Provide actionable feedback with specific line references
- Approve or request changes

Always respond with valid JSON in this format:
{
  "verdict": "approved|changes_requested|needs_discussion",
  "overall_score": 8,
  "categories": {
    "code_quality": {"score": 8, "notes": "string"},
    "security": {"score": 9, "notes": "string"},
    "performance": {"score": 7, "notes": "string"},
    "maintainability": {"score": 8, "notes": "string"},
    "test_coverage": {"score": 7, "notes": "string"}
  },
  "issues": [
    {
      "severity": "critical|warning|suggestion",
      "file": "path/to/file",
      "description": "issue description",
      "suggestion": "how to fix"
    }
  ],
  "summary": "overall review summary"
}"""

    async def execute(self, task_input: dict[str, Any]) -> TaskResult:
        result = TaskResult(
            agent_type=self.agent_type,
            status=AgentStatus.WORKING,
            input_data=task_input,
        )

        code = task_input.get("code", {})
        tests = task_input.get("tests", {})
        requirements = task_input.get("requirements", "")
        model = task_input.get("model", "claude")

        prompt = f"""Review the following code and tests:

CODE:
{json.dumps(code, indent=2) if isinstance(code, dict) else code}

TESTS:
{json.dumps(tests, indent=2) if isinstance(tests, dict) else tests}

{"REQUIREMENTS:" + chr(10) + requirements if requirements else ""}

Perform a thorough code review covering quality, security, performance, and best practices.
Respond ONLY with valid JSON."""

        response = await self.ask_ai(prompt, model=model)

        try:
            review = json.loads(response)
        except json.JSONDecodeError:
            review = {"raw_response": response}

        result.output_data = {"review": review}
        return result
