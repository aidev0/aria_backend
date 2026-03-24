from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from agents import (
    AgentType,
    CodeReviewerAgent,
    DeployerAgent,
    DeveloperAgent,
    PlannerAgent,
    ReporterAgent,
    TesterAgent,
)
from services import (
    AIService,
    ElevenLabsService,
    StreamService,
    WhatsAppService,
    WhisperService,
)

# ── Global state ──────────────────────────────────────────────────────────────

glasses_connections: set[WebSocket] = set()
view_connections: set[WebSocket] = set()

ai_service = AIService()
whisper_service = WhisperService()
elevenlabs_service = ElevenLabsService()
whatsapp_service = WhatsAppService()
stream_service = StreamService()

# Agents
planner = PlannerAgent(ai_service)
developer = DeveloperAgent(ai_service)
tester = TesterAgent(ai_service)
code_reviewer = CodeReviewerAgent(ai_service)
deployer = DeployerAgent(ai_service)
reporter = ReporterAgent(ai_service, whatsapp_service, elevenlabs_service)

AGENTS = {
    AgentType.PLANNER: planner,
    AgentType.DEVELOPER: developer,
    AgentType.TESTER: tester,
    AgentType.CODE_REVIEWER: code_reviewer,
    AgentType.DEPLOYER: deployer,
    AgentType.REPORTER: reporter,
}


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Wire up stream service → broadcast AI responses to glasses & viewers
    async def broadcast_ai_response(response: dict):
        msg = json.dumps(response)
        for ws in glasses_connections:
            try:
                await ws.send_text(msg)
            except Exception:
                pass
        for ws in view_connections:
            try:
                await ws.send_text(msg)
            except Exception:
                pass

    stream_service.subscribe_ai_response(broadcast_ai_response)

    # Wire up audio stream → Whisper STT
    async def handle_audio(chunk):
        whisper_service.accumulate_audio(chunk.data)
        transcription = await whisper_service.transcribe_stream(min_duration_ms=3000)
        if transcription:
            await broadcast_ai_response({
                "type": "transcription",
                "text": transcription["text"],
                "segments": transcription.get("segments", []),
            })

    stream_service.subscribe_audio(handle_audio)
    yield


app = FastAPI(
    title="Aria AI Backend",
    description="AI development agents for Aria Glasses — plan, develop, test, review, deploy, report",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ────────────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    task_input: dict[str, Any]
    model: str = "claude"


class PipelineRequest(BaseModel):
    requirement: str
    context: str = ""
    model: str = "claude"
    whatsapp_phone: str | None = None
    deploy_target: str = "docker"


class TTSRequest(BaseModel):
    text: str
    voice_id: str | None = None


class TranscribeRequest(BaseModel):
    audio_base64: str
    language: str | None = None


# ── REST API: Status ──────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "ok",
        "name": "Aria AI Backend",
        "version": "2.0.0",
        "glasses_connected": len(glasses_connections),
        "viewers_connected": len(view_connections),
        "agents": {k.value: v.status.value for k, v in AGENTS.items()},
        "streams": stream_service.get_stats(),
    }


# ── REST API: Agents ──────────────────────────────────────────────────────────

@app.get("/agents")
async def list_agents():
    return {
        "agents": [agent.get_status() for agent in AGENTS.values()]
    }


@app.get("/agents/{agent_type}")
async def get_agent(agent_type: str):
    try:
        at = AgentType(agent_type)
    except ValueError:
        return {"error": f"Unknown agent: {agent_type}"}
    return AGENTS[at].get_status()


@app.post("/agents/{agent_type}/run")
async def run_agent(agent_type: str, request: AgentRequest):
    try:
        at = AgentType(agent_type)
    except ValueError:
        return {"error": f"Unknown agent: {agent_type}"}

    agent = AGENTS[at]
    request.task_input["model"] = request.model
    result = await agent.run(request.task_input, model=request.model)
    return result.model_dump()


# ── REST API: Full Pipeline ───────────────────────────────────────────────────

@app.post("/pipeline/run")
async def run_pipeline(request: PipelineRequest):
    """Run the full development pipeline: plan → develop → test → review → deploy → report."""
    pipeline_results: dict[str, Any] = {}

    # 1. Plan
    plan_result = await planner.run({
        "requirement": request.requirement,
        "context": request.context,
        "model": request.model,
    })
    pipeline_results["planning"] = plan_result.output_data

    # Notify glasses/viewers of progress
    await stream_service.push_ai_response(
        text="Planning complete. Starting development...",
        display_data={"stage": "planning", "status": "complete"},
    )

    # 2. Develop (first task from plan)
    plan = plan_result.output_data.get("plan", {})
    tasks = plan.get("tasks", [])
    dev_results = []
    for task in tasks[:5]:  # Limit to first 5 tasks
        dev_result = await developer.run({
            "task": task,
            "plan_context": json.dumps(plan),
            "model": request.model,
        })
        dev_results.append(dev_result.output_data)

    pipeline_results["development"] = dev_results

    await stream_service.push_ai_response(
        text="Development complete. Running tests...",
        display_data={"stage": "development", "status": "complete"},
    )

    # 3. Test
    test_result = await tester.run({
        "code": dev_results,
        "task_spec": plan,
        "acceptance_criteria": [
            c for t in tasks for c in t.get("acceptance_criteria", [])
        ],
        "model": request.model,
    })
    pipeline_results["testing"] = test_result.output_data

    await stream_service.push_ai_response(
        text="Testing complete. Running code review...",
        display_data={"stage": "testing", "status": "complete"},
    )

    # 4. Code Review
    review_result = await code_reviewer.run({
        "code": dev_results,
        "tests": test_result.output_data,
        "requirements": request.requirement,
        "model": request.model,
    })
    pipeline_results["code_review"] = review_result.output_data

    await stream_service.push_ai_response(
        text="Code review complete. Preparing deployment...",
        display_data={"stage": "code_review", "status": "complete"},
    )

    # 5. Deploy
    deploy_result = await deployer.run({
        "project": plan,
        "code": dev_results,
        "review": review_result.output_data,
        "target": request.deploy_target,
        "model": request.model,
    })
    pipeline_results["deployment"] = deploy_result.output_data

    # 6. Report
    report_result = await reporter.run({
        "pipeline_results": pipeline_results,
        "whatsapp_phone": request.whatsapp_phone,
        "model": request.model,
    })

    # Send voice memo to glasses
    voice_audio = report_result.output_data.get("voice_audio")
    report_text = report_result.output_data.get("report", {}).get("voice_memo", "Pipeline complete.")
    await stream_service.push_ai_response(
        text=report_text,
        audio_data=voice_audio if isinstance(voice_audio, bytes) else None,
        display_data=report_result.output_data.get("report", {}).get("display_notification"),
    )

    return {
        "status": "complete",
        "pipeline_results": pipeline_results,
        "report": report_result.output_data,
    }


# ── REST API: Speech Services ─────────────────────────────────────────────────

@app.post("/speech/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using ElevenLabs."""
    audio = await elevenlabs_service.text_to_speech(
        text=request.text,
        voice_id=request.voice_id,
    )
    audio_b64 = base64.b64encode(audio).decode("utf-8")
    return {"audio_base64": audio_b64, "format": "mp3"}


@app.post("/speech/tts/stream")
async def text_to_speech_stream(request: TTSRequest):
    """Stream TTS audio to glasses speaker."""
    audio = await elevenlabs_service.text_to_speech(request.text, request.voice_id)
    await stream_service.push_ai_response(
        text=request.text,
        audio_data=audio,
    )
    return {"status": "streamed", "text": request.text}


@app.post("/speech/stt")
async def speech_to_text(request: TranscribeRequest):
    """Transcribe audio using OpenAI Whisper."""
    audio_bytes = base64.b64decode(request.audio_base64)
    result = await whisper_service.transcribe(
        audio_data=audio_bytes,
        language=request.language,
    )
    return result


# ── WebSocket: Glasses ────────────────────────────────────────────────────────

@app.websocket("/glasses")
async def glasses_endpoint(websocket: WebSocket):
    await websocket.accept()
    glasses_connections.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                continue

            msg_type = message.get("type", "")

            if msg_type == "frame":
                # Video frame from glasses camera
                frame_data = message.get("data", "")
                await stream_service.push_video_frame(frame_data)
                # Forward to viewers
                for view in view_connections:
                    try:
                        await view.send_text(data)
                    except Exception:
                        pass

            elif msg_type == "audio":
                # Audio chunk from glasses mic
                audio_b64 = message.get("data", "")
                audio_bytes = base64.b64decode(audio_b64)
                await stream_service.push_audio_chunk(audio_bytes)

            elif msg_type == "command":
                # Voice command (already transcribed by iOS app)
                command_text = message.get("text", "")
                if command_text:
                    # Run through pipeline
                    asyncio.create_task(
                        _handle_voice_command(command_text, message.get("model", "claude"))
                    )

            else:
                # Forward unknown messages to viewers
                for view in view_connections:
                    try:
                        await view.send_text(data)
                    except Exception:
                        pass

    except WebSocketDisconnect:
        glasses_connections.discard(websocket)


async def _handle_voice_command(command: str, model: str = "claude"):
    """Process a voice command through the appropriate agent or pipeline."""
    # Simple command routing
    cmd_lower = command.lower()

    if any(word in cmd_lower for word in ["build", "create", "make", "develop"]):
        result = await planner.run({"requirement": command, "model": model})
        text = json.dumps(result.output_data.get("plan", {}).get("summary", "Plan created."))
    elif any(word in cmd_lower for word in ["test", "check", "verify"]):
        result = await tester.run({"code": command, "model": model})
        text = "Tests generated."
    elif any(word in cmd_lower for word in ["review", "look at"]):
        result = await code_reviewer.run({"code": command, "model": model})
        text = "Code review complete."
    elif any(word in cmd_lower for word in ["deploy", "ship", "release", "launch"]):
        result = await deployer.run({"project": command, "model": model})
        text = "Deployment plan ready."
    elif any(word in cmd_lower for word in ["status", "report", "how"]):
        statuses = {k.value: v.status.value for k, v in AGENTS.items()}
        text = f"Agent statuses: {json.dumps(statuses)}"
    else:
        # Default: treat as a full pipeline request
        response = await ai_service.generate(
            prompt=command,
            system_prompt="You are Aria, an AI assistant integrated with AR glasses. Be concise and helpful.",
            model=model,
        )
        text = response

    # Generate TTS and push to glasses
    try:
        audio = await elevenlabs_service.text_to_speech(text[:500])
        await stream_service.push_ai_response(text=text, audio_data=audio)
    except Exception:
        await stream_service.push_ai_response(text=text)


# ── WebSocket: Web Viewer ─────────────────────────────────────────────────────

@app.websocket("/view")
async def view_endpoint(websocket: WebSocket):
    await websocket.accept()
    view_connections.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                # Forward raw messages to glasses
                for glasses in glasses_connections:
                    try:
                        await glasses.send_text(data)
                    except Exception:
                        pass
                continue

            msg_type = message.get("type", "")

            if msg_type == "command":
                # Command from web viewer to glasses
                for glasses in glasses_connections:
                    try:
                        await glasses.send_text(data)
                    except Exception:
                        pass
            elif msg_type == "agent_command":
                # Run agent from web UI
                agent_type = message.get("agent", "")
                task_input = message.get("task_input", {})
                model = message.get("model", "claude")
                try:
                    at = AgentType(agent_type)
                    agent = AGENTS[at]
                    task_input["model"] = model
                    result = await agent.run(task_input)
                    await websocket.send_text(json.dumps({
                        "type": "agent_result",
                        "agent": agent_type,
                        "result": result.model_dump(),
                    }, default=str))
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "agent_error",
                        "error": str(e),
                    }))
            else:
                # Forward to glasses
                for glasses in glasses_connections:
                    try:
                        await glasses.send_text(data)
                    except Exception:
                        pass

    except WebSocketDisconnect:
        view_connections.discard(websocket)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
