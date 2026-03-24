from __future__ import annotations

import asyncio
import json
import os
import tempfile
from enum import Enum
from typing import Any


class CLIType(str, Enum):
    CLAUDE = "claude"
    GEMINI = "gemini"
    CODEX = "codex"
    NONE = "none"


# CLI commands: npm i -g @anthropic-ai/claude-code | @google/gemini-cli | @openai/codex
CLI_COMMANDS = {
    CLIType.CLAUDE: "claude",
    CLIType.GEMINI: "gemini",
    CLIType.CODEX: "codex",
}


class CLIService:
    """Execute tasks via Claude Code, Gemini CLI, or Codex CLI."""

    def __init__(self, working_dir: str | None = None):
        self.working_dir = working_dir or os.getcwd()

    async def execute(
        self,
        cli: CLIType,
        prompt: str,
        working_dir: str | None = None,
        timeout: int = 300,
    ) -> dict[str, Any]:
        """Run a prompt through the specified CLI tool.

        Args:
            cli: Which CLI to use (claude, gemini, codex)
            prompt: The prompt/instruction to send
            working_dir: Working directory for the CLI
            timeout: Max seconds to wait

        Returns:
            Dict with stdout, stderr, return_code
        """
        if cli == CLIType.NONE:
            return {"error": "No CLI selected", "stdout": "", "stderr": "", "return_code": 1}

        cmd_name = CLI_COMMANDS[cli]
        cwd = working_dir or self.working_dir

        if cli == CLIType.CLAUDE:
            cmd = self._build_claude_cmd(prompt)
        elif cli == CLIType.GEMINI:
            cmd = self._build_gemini_cmd(prompt)
        elif cli == CLIType.CODEX:
            cmd = self._build_codex_cmd(prompt)
        else:
            return {"error": f"Unknown CLI: {cli}", "stdout": "", "stderr": "", "return_code": 1}

        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env={**os.environ, "CI": "1"},  # Non-interactive mode
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "cli": cli.value,
                    "stdout": "",
                    "stderr": f"CLI timed out after {timeout}s",
                    "return_code": -1,
                    "timed_out": True,
                }

            return {
                "cli": cli.value,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "return_code": process.returncode or 0,
            }

        except FileNotFoundError:
            return {
                "cli": cli.value,
                "stdout": "",
                "stderr": f"CLI '{cmd_name}' not found. Install: npm i -g {self._get_package(cli)}",
                "return_code": 1,
            }
        except Exception as e:
            return {
                "cli": cli.value,
                "stdout": "",
                "stderr": str(e),
                "return_code": 1,
            }

    def _build_claude_cmd(self, prompt: str) -> str:
        safe = prompt.replace("'", "'\\''")
        return f"claude -p '{safe}' --output-format json"

    def _build_gemini_cmd(self, prompt: str) -> str:
        safe = prompt.replace("'", "'\\''")
        return f"gemini -p '{safe}'"

    def _build_codex_cmd(self, prompt: str) -> str:
        safe = prompt.replace("'", "'\\''")
        return f"codex -q '{safe}' --json"

    def _get_package(self, cli: CLIType) -> str:
        packages = {
            CLIType.CLAUDE: "@anthropic-ai/claude-code",
            CLIType.GEMINI: "@google/gemini-cli",
            CLIType.CODEX: "@openai/codex",
        }
        return packages.get(cli, "unknown")

    @staticmethod
    def available_clis() -> list[dict[str, str]]:
        return [
            {"value": "claude", "label": "Claude Code", "package": "@anthropic-ai/claude-code"},
            {"value": "gemini", "label": "Gemini CLI", "package": "@google/gemini-cli"},
            {"value": "codex", "label": "Codex CLI", "package": "@openai/codex"},
            {"value": "none", "label": "API Only", "package": ""},
        ]
