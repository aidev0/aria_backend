from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass


@dataclass
class CommandResult:
    command: str
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool = False


class ShellExecutor:
    """Safe shell command executor for agent tools (build, test, deploy commands)."""

    ALLOWED_COMMANDS = {
        "npm", "npx", "node", "python", "python3", "pip", "pip3",
        "pytest", "jest", "git", "docker", "docker-compose",
        "curl", "ls", "cat", "mkdir", "cp", "mv", "echo",
        "make", "cargo", "go", "swift", "xcodebuild",
    }

    def __init__(self, working_dir: str | None = None, timeout: int = 120):
        self.working_dir = working_dir or os.getcwd()
        self.timeout = timeout

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
    ) -> CommandResult:
        """Execute a shell command safely.

        Args:
            command: Shell command to run
            working_dir: Working directory override
            timeout: Timeout in seconds override
        """
        cmd_name = command.split()[0] if command else ""
        if cmd_name not in self.ALLOWED_COMMANDS:
            return CommandResult(
                command=command,
                stdout="",
                stderr=f"Command '{cmd_name}' is not in the allowed commands list.",
                return_code=1,
            )

        cwd = working_dir or self.working_dir
        cmd_timeout = timeout or self.timeout

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=cmd_timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return CommandResult(
                    command=command,
                    stdout="",
                    stderr=f"Command timed out after {cmd_timeout}s",
                    return_code=-1,
                    timed_out=True,
                )

            return CommandResult(
                command=command,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                return_code=process.returncode or 0,
            )

        except Exception as e:
            return CommandResult(
                command=command,
                stdout="",
                stderr=str(e),
                return_code=1,
            )

    async def run_tests(self, project_dir: str, framework: str = "pytest") -> CommandResult:
        """Run tests for a project."""
        commands = {
            "pytest": "python -m pytest -v",
            "jest": "npx jest --verbose",
            "swift": "swift test",
        }
        cmd = commands.get(framework, f"{framework}")
        return await self.execute(cmd, working_dir=project_dir)

    async def build_project(self, project_dir: str, build_cmd: str = "npm run build") -> CommandResult:
        """Build a project."""
        return await self.execute(build_cmd, working_dir=project_dir)
