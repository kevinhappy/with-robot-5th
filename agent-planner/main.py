from __future__ import annotations

import base64
import json
import os
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from src.config import config
from src.executor import TaskExecutor
from src.graph import create_graph
from src.state import make_state

# Simulator endpoint (environment objects + action execution)
SIM_URL = "http://127.0.0.1:8800"

# Server configuration
HOST = "0.0.0.0"  # Listen on all network interfaces
PORT = 8900       # API server port
VERSION = "0.0.1"

_HERE = os.path.dirname(os.path.abspath(__file__))

# FastAPI application instance
app = FastAPI(
    title="LLM Agent API",
    description="Goal/Task decomposition planner",
    version=VERSION,
)

# Load environment variables from .env if present
load_dotenv()


# --- ElevenLabs (optional) -------------------------------------------------
# Speech endpoints only work when ELEVENLABS_API_KEY is configured; the server
# still boots without it.
elevenlabs_client = None
_elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
if _elevenlabs_key:
    try:
        from elevenlabs.client import ElevenLabs

        elevenlabs_client = ElevenLabs(api_key=_elevenlabs_key)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] ElevenLabs disabled: {exc}")


def _require_elevenlabs():
    if elevenlabs_client is None:
        raise RuntimeError(
            "ELEVENLABS_API_KEY is not set; /stt and /tts are disabled."
        )
    return elevenlabs_client


# --- LangGraph pipeline (lazy) -------------------------------------------------
# Building the graph constructs ChatOpenAI clients, which require OPENAI_API_KEY.
# Defer it to the first request so the server can start without a key.
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = create_graph(config)
    return _graph


@app.get("/")
def get_ui() -> HTMLResponse:
    """Serve the web UI."""
    with open(os.path.join(_HERE, "ui.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/llm_command")
def llm_command(request: dict):
    """
    Receives natural language commands and generates/executes robot control code.

    Request format:
        {"command": "Organize the objects to the bowls according to their colors"}

    Response format:
        {"status": "success", "user_command": "...", "generated_code": "..."}
    """
    try:
        user_command = request.get("command", "")

        if not user_command:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"status": "error", "message": "No command provided"},
            )

        state = make_state(user_query=user_command, config=config, url=SIM_URL)

        final_state = _get_graph().invoke(state)
        task_outputs = final_state["tasks"]["task_outputs"]

        print("*" * 40)
        print(task_outputs)
        print("*" * 40)

        results = TaskExecutor(url=SIM_URL).execute(task_outputs)

        generated_code = json.dumps(results, ensure_ascii=False, indent=2)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "user_command": user_command,
                "generated_code": generated_code,
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        )


@app.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """Convert speech to text using ElevenLabs STT (Korean)."""
    try:
        client = _require_elevenlabs()
        audio_bytes = await audio.read()

        transcription = client.speech_to_text.convert(
            file=audio_bytes,
            model_id="scribe_v1",
            language_code="ko",
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "success", "text": transcription.text},
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        )


@app.post("/tts")
async def text_to_speech(request: dict):
    """Convert text to speech using ElevenLabs TTS. Returns base64 mp3."""
    try:
        client = _require_elevenlabs()
        text = request.get("text", "")

        if not text:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"status": "error", "message": "No text provided"},
            )

        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id="XB0fDUnXU5powFXDhCwa",  # Charlotte - multilingual voice
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )

        audio_bytes = b"".join(audio_generator)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "success", "audio": audio_base64},
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LLM Agent API")
    print("=" * 60)
    print(f"Server:   http://{HOST}:{PORT}")
    print(f"API docs: http://{HOST}:{PORT}/docs")
    print(f"Simulator expected at: {SIM_URL}")
    if elevenlabs_client is None:
        print("ElevenLabs: disabled (no ELEVENLABS_API_KEY)")
    print("=" * 60 + "\n")

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
