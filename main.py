from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from workos import WorkOSClient

load_dotenv()

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
    CLIService,
    Database,
    GCSService,
    ElevenLabsService,
    StreamService,
    WhatsAppService,
    WhisperService,
)
from services.ai_service import MODELS
from services.cli_service import CLIType

# ── Global state ──────────────────────────────────────────────────────────────

glasses_connections: set[WebSocket] = set()
view_connections: set[WebSocket] = set()

# ── WorkOS Auth ──────────────────────────────────────────────────────────────

workos_client = WorkOSClient(
    api_key=os.getenv("WORKOS_API_KEY", ""),
    client_id=os.getenv("WORKOS_CLIENT_ID", ""),
)

# Map token -> user info for authenticated WebSocket connections
authenticated_users: dict[str, dict] = {}

db = Database()
ai_service = AIService()
cli_service = CLIService()
gcs_service = GCSService()
whisper_service = WhisperService()
elevenlabs_service = ElevenLabsService()
whatsapp_service = WhatsAppService()
stream_service = StreamService()

# Agents — each gets both ai_service and cli_service
planner = PlannerAgent(ai_service, cli_service)
developer = DeveloperAgent(ai_service, cli_service)
tester = TesterAgent(ai_service, cli_service)
code_reviewer = CodeReviewerAgent(ai_service, cli_service)
deployer = DeployerAgent(ai_service, cli_service)
reporter = ReporterAgent(ai_service, whatsapp_service, elevenlabs_service)
reporter.cli_service = cli_service

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
    # Connect MongoDB
    await db.connect()

    # Load saved agent configs from MongoDB
    saved_configs = await db.get_agent_configs()
    for cfg in saved_configs:
        agent_type_str = cfg.get("agent_type")
        try:
            at = AgentType(agent_type_str)
            if at in AGENTS:
                AGENTS[at].configure(cfg)
        except (ValueError, KeyError):
            pass

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

    # Wire up audio stream → Whisper STT → GCS + MongoDB persistence
    async def handle_audio(chunk):
        whisper_service.accumulate_audio(chunk.data)
        transcription = await whisper_service.transcribe_stream(min_duration_ms=3000)
        if transcription:
            session_id = uuid.uuid4().hex
            wav_data = transcription.pop("wav_data", None)
            duration_ms = transcription.pop("duration_ms", 0)

            # Upload audio to Google Cloud Storage (best-effort, non-blocking)
            gcs_info = {}
            if wav_data:
                try:
                    gcs_info = await gcs_service.upload_audio(
                        wav_data, session_id=session_id
                    )
                except Exception as e:
                    print(f"[GCS] Upload failed (non-fatal): {e}")

            # Persist voice session to MongoDB
            try:
                await db.save_voice_session({
                    "session_id": session_id,
                    "transcription": transcription["text"],
                    "segments": transcription.get("segments", []),
                    "language": transcription.get("language"),
                    "duration_ms": duration_ms,
                    "audio_gcs_uri": gcs_info.get("gcs_uri"),
                    "audio_blob_name": gcs_info.get("blob_name"),
                    "audio_size_bytes": gcs_info.get("size_bytes"),
                })
            except Exception as e:
                print(f"[DB] Save voice session failed (non-fatal): {e}")

            # Broadcast transcription back to iOS app + web viewers
            await broadcast_ai_response({
                "type": "transcription",
                "session_id": session_id,
                "text": transcription["text"],
                "segments": transcription.get("segments", []),
            })

    stream_service.subscribe_audio(handle_audio)

    yield

    await db.disconnect()


app = FastAPI(
    title="Aria AI Backend",
    description="AI development agents for Aria Glasses — plan, develop, test, review, deploy, report",
    version="3.0.0",
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
    model: str | None = None  # Override per-request, else uses agent config


class AgentConfigRequest(BaseModel):
    model: str = "claude"       # claude | gemini | openai
    cli: str = "none"           # claude | gemini | codex | none
    use_cli: bool = False


class PipelineRequest(BaseModel):
    requirement: str
    context: str = ""
    model: str | None = None    # Override all agents, else each uses its own config
    whatsapp_phone: str | None = None
    deploy_target: str = "docker"


class TTSRequest(BaseModel):
    text: str
    voice_id: str | None = None


class TranscribeRequest(BaseModel):
    audio_base64: str
    language: str | None = None


class AuthCodeRequest(BaseModel):
    code: str
    redirect_uri: str | None = None


# ── REST API: Auth ───────────────────────────────────────────────────────────

@app.post("/auth/token")
async def exchange_auth_code(request: AuthCodeRequest):
    """Exchange a WorkOS authorization code for user profile + access token (used by iOS app)."""
    try:
        auth_response = workos_client.user_management.authenticate_with_code(
            code=request.code,
            session=None,
        )
        user = auth_response.user
        access_token = auth_response.access_token

        user_info = {
            "id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
        }

        # Cache for WebSocket auth
        authenticated_users[access_token] = user_info

        return {
            "access_token": access_token,
            "user": user_info,
        }
    except Exception as e:
        return {"error": str(e)}


# ── REST API: Status ──────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "ok",
        "name": "Aria AI Backend",
        "version": "3.0.0",
        "models": MODELS,
        "glasses_connected": len(glasses_connections),
        "viewers_connected": len(view_connections),
        "agents": {k.value: v.get_status() for k, v in AGENTS.items()},
        "streams": stream_service.get_stats(),
    }


# ── REST API: Models & CLIs ──────────────────────────────────────────────────

@app.get("/models")
async def list_models():
    return {
        "models": [
            {"value": "claude", "label": "Claude Opus 4.6", "model_id": MODELS["claude"]},
            {"value": "gemini", "label": "Gemini 3.1 Pro", "model_id": MODELS["gemini"]},
            {"value": "openai", "label": "GPT-5.4", "model_id": MODELS["openai"]},
        ]
    }


@app.get("/clis")
async def list_clis():
    return {"clis": CLIService.available_clis()}


# ── REST API: Agent Config ────────────────────────────────────────────────────

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


@app.put("/agents/{agent_type}/config")
async def configure_agent(agent_type: str, request: AgentConfigRequest):
    """Set model and CLI for a specific agent. Persists to MongoDB."""
    try:
        at = AgentType(agent_type)
    except ValueError:
        return {"error": f"Unknown agent: {agent_type}"}

    config = request.model_dump()
    AGENTS[at].configure(config)

    # Persist to MongoDB
    await db.set_agent_config(agent_type, config)

    return {
        "agent_type": agent_type,
        "config": AGENTS[at].config.model_dump(),
        "saved": True,
    }


@app.get("/agents/configs/all")
async def get_all_agent_configs():
    """Get saved configs for all agents from MongoDB."""
    configs = await db.get_agent_configs()
    return {"configs": configs}


# ── REST API: Run Agent ───────────────────────────────────────────────────────

@app.post("/agents/{agent_type}/run")
async def run_agent(agent_type: str, request: AgentRequest):
    try:
        at = AgentType(agent_type)
    except ValueError:
        return {"error": f"Unknown agent: {agent_type}"}

    agent = AGENTS[at]
    result = await agent.run(request.task_input, model=request.model)

    # Save to MongoDB
    await db.save_task_result(result.model_dump())

    return result.model_dump()


# ── REST API: Full Pipeline ───────────────────────────────────────────────────

@app.post("/pipeline/run")
async def run_pipeline(request: PipelineRequest):
    """Run the full development pipeline: plan → develop → test → review → deploy → report.
    Each agent uses its own configured model/CLI unless overridden in the request."""
    pipeline_results: dict[str, Any] = {}

    # 1. Plan
    plan_result = await planner.run({
        "requirement": request.requirement,
        "context": request.context,
    }, model=request.model)
    pipeline_results["planning"] = plan_result.output_data

    await stream_service.push_ai_response(
        text="Planning complete. Starting development...",
        display_data={"stage": "planning", "status": "complete"},
    )

    # 2. Develop
    plan = plan_result.output_data.get("plan", {})
    tasks = plan.get("tasks", [])
    dev_results = []
    for task in tasks[:5]:
        dev_result = await developer.run({
            "task": task,
            "plan_context": json.dumps(plan),
        }, model=request.model)
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
    }, model=request.model)
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
    }, model=request.model)
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
    }, model=request.model)
    pipeline_results["deployment"] = deploy_result.output_data

    # 6. Report
    report_result = await reporter.run({
        "pipeline_results": pipeline_results,
        "whatsapp_phone": request.whatsapp_phone,
    }, model=request.model)

    voice_audio = report_result.output_data.get("voice_audio")
    report_text = report_result.output_data.get("report", {}).get("voice_memo", "Pipeline complete.")
    await stream_service.push_ai_response(
        text=report_text,
        audio_data=voice_audio if isinstance(voice_audio, bytes) else None,
        display_data=report_result.output_data.get("report", {}).get("display_notification"),
    )

    # Save to MongoDB
    await db.save_pipeline_run({
        "requirement": request.requirement,
        "model_override": request.model,
        "results": pipeline_results,
        "report": report_result.output_data,
    })

    return {
        "status": "complete",
        "pipeline_results": pipeline_results,
        "report": report_result.output_data,
    }


@app.get("/pipeline/history")
async def pipeline_history():
    """Get recent pipeline runs from MongoDB."""
    runs = await db.get_pipeline_runs()
    return {"runs": runs}


# ── REST API: Speech Services ─────────────────────────────────────────────────

@app.post("/speech/tts")
async def text_to_speech(request: TTSRequest):
    audio = await elevenlabs_service.text_to_speech(
        text=request.text, voice_id=request.voice_id,
    )
    return {"audio_base64": base64.b64encode(audio).decode("utf-8"), "format": "mp3"}


@app.post("/speech/tts/stream")
async def text_to_speech_stream(request: TTSRequest):
    audio = await elevenlabs_service.text_to_speech(request.text, request.voice_id)
    await stream_service.push_ai_response(text=request.text, audio_data=audio)
    return {"status": "streamed", "text": request.text}


@app.post("/speech/stt")
async def speech_to_text(request: TranscribeRequest):
    audio_bytes = base64.b64decode(request.audio_base64)
    return await whisper_service.transcribe(audio_data=audio_bytes, language=request.language)


# ── REST API: Voice Sessions ──────────────────────────────────────────────────

@app.get("/voice/sessions")
async def list_voice_sessions(limit: int = 50):
    """Get recent voice transcription sessions from MongoDB."""
    sessions = await db.get_voice_sessions(limit=limit)
    return {"sessions": sessions}


@app.get("/voice/sessions/{session_id}")
async def get_voice_session(session_id: str):
    """Get a single voice session by ID."""
    session = await db.get_voice_session(session_id)
    if not session:
        return {"error": "Session not found"}
    return session


@app.get("/voice/sessions/{session_id}/audio")
async def get_voice_session_audio_url(session_id: str):
    """Get a signed URL for the audio recording of a voice session."""
    session = await db.get_voice_session(session_id)
    if not session:
        return {"error": "Session not found"}

    blob_name = session.get("audio_blob_name")
    if not blob_name:
        return {"error": "No audio recording for this session"}

    try:
        url = gcs_service.get_signed_url(blob_name)
        return {"url": url, "session_id": session_id}
    except Exception as e:
        return {"error": f"Could not generate URL: {e}"}


# ── WebSocket: Glasses ────────────────────────────────────────────────────────

@app.websocket("/glasses")
async def glasses_endpoint(websocket: WebSocket, token: str = Query(default="")):
    await websocket.accept()
    # Attach user info if token provided
    user = authenticated_users.get(token)
    if user:
        websocket.state.user = user
    glasses_connections.add(websocket)
    print(f"[Glasses] Connected. Total: {len(glasses_connections)}")

    # Notify viewers that glasses connected
    status_msg = json.dumps({"type": "glasses_status", "connected": True, "count": len(glasses_connections)})
    for view in view_connections:
        try:
            await view.send_text(status_msg)
        except Exception:
            pass

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                continue

            msg_type = message.get("type", "")

            if msg_type == "frame":
                frame_data = message.get("data", "")
                await stream_service.push_video_frame(frame_data)
                for view in view_connections:
                    try:
                        await view.send_text(data)
                    except Exception:
                        pass

            elif msg_type == "audio":
                audio_b64 = message.get("data", "")
                audio_bytes = base64.b64decode(audio_b64)
                print(f"[Audio] Received {len(audio_bytes)} bytes")
                await stream_service.push_audio_chunk(audio_bytes)

            elif msg_type == "transcription":
                # Apple Speech or other on-device STT — save to MongoDB and broadcast
                text = message.get("text", "")
                source = message.get("source", "unknown")
                if text.strip():
                    session_id = uuid.uuid4().hex
                    try:
                        await db.save_voice_session({
                            "session_id": session_id,
                            "transcription": text,
                            "source": source,
                            "language": "en",
                            "duration_ms": 0,
                        })
                    except Exception as e:
                        print(f"[DB] Save voice session failed (non-fatal): {e}")

                    # Broadcast to web viewers
                    broadcast_msg = json.dumps({
                        "type": "transcription",
                        "session_id": session_id,
                        "text": text,
                        "source": source,
                    })
                    for view in view_connections:
                        try:
                            await view.send_text(broadcast_msg)
                        except Exception:
                            pass

            elif msg_type == "command":
                command_text = message.get("text", "")
                if command_text:
                    asyncio.create_task(
                        _handle_voice_command(command_text, message.get("model"))
                    )

            else:
                for view in view_connections:
                    try:
                        await view.send_text(data)
                    except Exception:
                        pass

    except WebSocketDisconnect:
        glasses_connections.discard(websocket)
        print(f"[Glasses] Disconnected. Total: {len(glasses_connections)}")

        # Notify viewers that glasses disconnected
        status_msg = json.dumps({"type": "glasses_status", "connected": len(glasses_connections) > 0, "count": len(glasses_connections)})
        for view in view_connections:
            try:
                await view.send_text(status_msg)
            except Exception:
                pass


async def _handle_voice_command(command: str, model: str | None = None):
    """Process a voice command through the appropriate agent or pipeline."""
    cmd_lower = command.lower()

    if any(w in cmd_lower for w in ("build", "create", "make", "develop")):
        result = await planner.run({"requirement": command}, model=model)
        text = json.dumps(result.output_data.get("plan", {}).get("summary", "Plan created."))
    elif any(w in cmd_lower for w in ("test", "check", "verify")):
        result = await tester.run({"code": command}, model=model)
        text = "Tests generated."
    elif any(w in cmd_lower for w in ("review", "look at")):
        result = await code_reviewer.run({"code": command}, model=model)
        text = "Code review complete."
    elif any(w in cmd_lower for w in ("deploy", "ship", "release", "launch")):
        result = await deployer.run({"project": command}, model=model)
        text = "Deployment plan ready."
    elif any(w in cmd_lower for w in ("status", "report", "how")):
        statuses = {k.value: v.status.value for k, v in AGENTS.items()}
        text = f"Agent statuses: {json.dumps(statuses)}"
    else:
        response = await ai_service.generate(
            prompt=command,
            system_prompt="You are Aria, an AI assistant integrated with AR glasses. Be concise and helpful.",
            model=model or "claude",
        )
        text = response

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
                for glasses in glasses_connections:
                    try:
                        await glasses.send_text(data)
                    except Exception:
                        pass
                continue

            msg_type = message.get("type", "")

            if msg_type == "command":
                for glasses in glasses_connections:
                    try:
                        await glasses.send_text(data)
                    except Exception:
                        pass
            elif msg_type == "agent_command":
                agent_type = message.get("agent", "")
                task_input = message.get("task_input", {})
                model = message.get("model")
                try:
                    at = AgentType(agent_type)
                    result = await AGENTS[at].run(task_input, model=model)
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
