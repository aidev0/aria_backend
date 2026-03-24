from __future__ import annotations

import json
from typing import Any

from .base_agent import AgentStatus, AgentType, BaseAgent, TaskResult


class DeployerAgent(BaseAgent):
    agent_type = AgentType.DEPLOYER
    name = "Deployer"
    description = "Handles build, deployment, and release pipelines for shipping code to production."

    def get_system_prompt(self) -> str:
        return """You are Aria's Deployer Agent — a DevOps and deployment automation expert.

Your role:
- Generate deployment configurations (Docker, CI/CD, cloud configs)
- Create build scripts and deployment pipelines
- Handle environment configuration and secrets management
- Support multiple deployment targets (Vercel, AWS, GCP, Railway, Fly.io)
- Validate deployment readiness
- Generate rollback plans

Always respond with valid JSON in this format:
{
  "deployment_plan": {
    "target": "vercel|aws|gcp|railway|fly|docker",
    "strategy": "rolling|blue_green|canary|direct",
    "pre_deploy_checks": ["check 1", "check 2"],
    "steps": [
      {
        "order": 1,
        "action": "description",
        "command": "shell command or config",
        "critical": true
      }
    ]
  },
  "config_files": [
    {
      "path": "Dockerfile|docker-compose.yml|vercel.json|etc",
      "content": "full file content"
    }
  ],
  "environment_variables": [
    {"name": "VAR_NAME", "description": "what this var does", "required": true}
  ],
  "rollback_plan": "steps to rollback if deployment fails",
  "estimated_downtime": "zero|seconds|minutes"
}"""

    async def execute(self, task_input: dict[str, Any]) -> TaskResult:
        result = TaskResult(
            agent_type=self.agent_type,
            status=AgentStatus.WORKING,
            input_data=task_input,
        )

        project = task_input.get("project", {})
        code = task_input.get("code", {})
        review = task_input.get("review", {})
        target = task_input.get("target", "docker")
        model = task_input.get("model", "claude")

        prompt = f"""Create a deployment plan for the following project:

PROJECT:
{json.dumps(project, indent=2) if isinstance(project, dict) else project}

CODE STRUCTURE:
{json.dumps(code, indent=2) if isinstance(code, dict) else code}

CODE REVIEW STATUS:
{json.dumps(review, indent=2) if isinstance(review, dict) else review}

TARGET PLATFORM: {target}

Generate a complete deployment configuration and plan.
Respond ONLY with valid JSON."""

        response = await self.ask_ai(prompt, model=model)

        try:
            deploy_plan = json.loads(response)
        except json.JSONDecodeError:
            deploy_plan = {"raw_response": response}

        result.output_data = {"deployment": deploy_plan}
        return result
