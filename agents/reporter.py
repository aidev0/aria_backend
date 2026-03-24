from __future__ import annotations

import json
from typing import Any

from .base_agent import AgentStatus, AgentType, BaseAgent, TaskResult


class ReporterAgent(BaseAgent):
    agent_type = AgentType.REPORTER
    name = "Reporter"
    description = "Aggregates pipeline results and delivers reports via WhatsApp, voice memos (ElevenLabs), and glasses display."

    def __init__(self, ai_service, whatsapp_service=None, elevenlabs_service=None):
        super().__init__(ai_service)
        self.whatsapp_service = whatsapp_service
        self.elevenlabs_service = elevenlabs_service

    def get_system_prompt(self) -> str:
        return """You are Aria's Reporter Agent — a concise technical communicator.

Your role:
- Aggregate results from all pipeline stages (plan, develop, test, review, deploy)
- Generate clear, concise status reports
- Create voice-friendly summaries for glasses whisper notifications
- Format WhatsApp-friendly messages with key highlights
- Produce visual notification data for glasses display

Always respond with valid JSON in this format:
{
  "summary": "one-line overall status",
  "status": "success|partial|failed",
  "voice_memo": "natural, conversational summary for text-to-speech (2-3 sentences max)",
  "whatsapp_message": "formatted message with emojis for WhatsApp",
  "display_notification": {
    "title": "short title for glasses display",
    "body": "brief body text",
    "icon": "success|warning|error|info",
    "progress": 100
  },
  "details": {
    "planning": {"status": "done|pending|failed", "summary": "string"},
    "development": {"status": "done|pending|failed", "summary": "string", "files_created": 0},
    "testing": {"status": "done|pending|failed", "summary": "string", "tests_passed": 0, "tests_failed": 0},
    "code_review": {"status": "done|pending|failed", "summary": "string", "score": 0},
    "deployment": {"status": "done|pending|failed", "summary": "string", "url": ""}
  }
}"""

    async def execute(self, task_input: dict[str, Any]) -> TaskResult:
        result = TaskResult(
            agent_type=self.agent_type,
            status=AgentStatus.WORKING,
            input_data=task_input,
        )

        pipeline_results = task_input.get("pipeline_results", {})
        model = task_input.get("model", "claude")

        prompt = f"""Generate a comprehensive report for the following development pipeline results:

PIPELINE RESULTS:
{json.dumps(pipeline_results, indent=2)}

Create a clear, concise report with:
1. A voice memo suitable for whispering through AR glasses (conversational, 2-3 sentences)
2. A WhatsApp message with key highlights
3. A glasses display notification
4. Detailed status breakdown

Respond ONLY with valid JSON."""

        response = await self.ask_ai(prompt, model=model)

        try:
            report = json.loads(response)
        except json.JSONDecodeError:
            report = {"raw_response": response}

        # Send WhatsApp message if service is available
        if self.whatsapp_service and report.get("whatsapp_message"):
            phone = task_input.get("whatsapp_phone")
            if phone:
                try:
                    await self.whatsapp_service.send_message(
                        to=phone,
                        message=report["whatsapp_message"],
                    )
                    report["whatsapp_sent"] = True
                except Exception as e:
                    report["whatsapp_error"] = str(e)

        # Generate voice memo audio if ElevenLabs service is available
        if self.elevenlabs_service and report.get("voice_memo"):
            try:
                audio_data = await self.elevenlabs_service.text_to_speech(
                    text=report["voice_memo"],
                )
                report["voice_audio_available"] = True
                result.output_data["voice_audio"] = audio_data
            except Exception as e:
                report["voice_error"] = str(e)

        result.output_data = {"report": report}
        return result
