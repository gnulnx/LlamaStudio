"""
LLamaStudio - FastAPI backend with HTMX frontend.
A desktop-like chat interface for llama.cpp.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .chat import chat, validate_audio_attachments, validate_image_attachments
from .config import settings
from .config_store import config_loader
from .logger import logger
from .server_manager import server
from .speech_manager import SpeechError, speech

app = FastAPI(title="LLamaStudio")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
static_dir = Path(__file__).parent / "static"

# Mount static files
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Helper functions for model settings persistence
def load_model_settings() -> dict:
    return config_loader.get_model_settings_registry()


def save_model_settings(all_settings: dict):
    for model_path, profile_settings in all_settings.items():
        if isinstance(profile_settings, dict):
            config_loader.save_model_profile(model_path, profile_settings)


def _get_scanned_model(model_path: str):
    """Return an exact scanned model match, preventing arbitrary model-path operations."""
    from .model_manager import get_models

    requested = Path(model_path).expanduser().resolve()
    return next(
        (model for model in get_models() if Path(model.path).expanduser().resolve() == requested),
        None,
    )


# ─── Page routes ──────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    conversations = chat.list_conversations()
    status = server.get_status()
    from .model_manager import get_models

    models_available = bool(get_models())
    query_view = request.query_params.get("view")
    valid_views = {"chat", "discover", "models", "logs"}
    if query_view in valid_views:
        launch_view = query_view
    else:
        launch_view = config_loader.get_launch_view(
            model_loaded=status.get("running", False),
            models_available=models_available,
            consume_first_launch=True,
        )
    chat_defaults = config_loader.get_chat_defaults()
    return templates.TemplateResponse(
        request,
        name="index.html",
        context={
            "request": request,
            "conversations": conversations,
            "server_running": status.get("running", False),
            "server_loading": status.get("is_loading", False),
            "current_model": status.get("current_model"),
            "current_model_name": status.get("current_model_name"),
            "launch_view": launch_view,
            "system_prompt": chat_defaults["system_prompt"],
            "temperature": chat_defaults["temperature"],
            "top_p": chat_defaults["top_p"],
            "max_tokens": chat_defaults["max_tokens"],
        },
    )


# ─── Server & Model management ────────────────────────────────


@app.get("/api/server/status")
async def server_status():
    return server.get_status()


@app.get("/api/gpu")
async def get_gpu():
    """Retrieve primary GPU name and VRAM (in GB)."""
    from .gpu_utils import get_gpu_info

    return get_gpu_info()


@app.get("/api/server/logs")
async def server_logs(type: str = "llama", lines: int = 50):
    log_name = "app.log" if type == "app" else "server.log"
    log_file = Path(settings.LOG_DIR) / log_name
    if not log_file.exists():
        return {"logs": []}
    try:
        with open(log_file, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        return {"logs": all_lines[-lines:]}
    except Exception as e:
        logger.error(f"Error reading {log_name}: {e}")
        return {"logs": [f"Error reading log file: {e}\n"]}


# ─── Model discovery & Settings ───────────────────────────────


@app.get("/api/models")
async def list_models():
    """Scan model directories for GGUF files."""
    from .model_manager import get_models

    models = get_models()
    return {
        "models": [
            {
                "path": m.path,
                "name": m.name,
                "size": m.size,
                "size_human": m.size_human,
                "quant": m.quant,
                "is_multimodal": m.is_multimodal,
                "mmproj_path": m.mmproj_path,
                "is_loaded": m.path == server._current_model,
            }
            for m in models
        ]
    }


@app.post("/api/models/load")
async def load_model(request: Request):
    """Load a specific model with customized parameters."""
    body = await request.json()
    model_path = body.get("path")
    model_params = body.get("settings", {})
    if not model_path:
        raise HTTPException(400, "Model path is required")

    config_loader.save_model_profile(model_path, model_params)
    result = server.load_model(model_path, model_params)
    if not result:
        raise HTTPException(500, "Failed to load model. Check server logs.")

    return {"status": "ok", "model": model_path, "running": server.is_running}


@app.post("/api/models/reload-current")
async def reload_current_model():
    """Reload the active model so a newly downloaded projector is attached."""
    model_path = server._current_model
    if not model_path:
        raise HTTPException(409, "No model is currently loaded.")

    model_params = dict(server._current_params)
    model_params.pop("mmproj", None)
    if not server.load_model(model_path, model_params):
        raise HTTPException(500, "Failed to reload the model with its vision projector.")

    return {
        "status": "ok",
        "model": model_path,
        "multimodal": server.supports_multimodal(),
        "audio": server.supports_audio(),
    }


@app.get("/api/models/projector")
async def get_model_projector(model_path: str):
    """Resolve a local or downloadable projector for a scanned model."""
    model = _get_scanned_model(model_path)
    if model is None:
        raise HTTPException(404, "Model is not present in a configured model directory.")

    from .model_manager import resolve_model_projector

    return await resolve_model_projector(model.path)


@app.post("/api/models/eject")
async def eject_model():
    """Eject the currently loaded model."""
    server.eject_model()
    return {"status": "ok", "running": False}


# ─── Local speech-to-text ─────────────────────────────────────


@app.get("/api/speech/status")
async def speech_status():
    """Return whisper.cpp installation and process state."""
    return speech.get_status()


@app.post("/api/speech/load")
async def load_speech(request: Request):
    """Load the configured Whisper model into the managed speech server."""
    body = await request.json()
    model = body.get("model")
    use_gpu = body.get("use_gpu")
    if use_gpu is not None and not isinstance(use_gpu, bool):
        raise HTTPException(400, "use_gpu must be a boolean.")
    try:
        return await asyncio.to_thread(speech.start, model=model, use_gpu=use_gpu)
    except SpeechError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/api/speech/eject")
async def eject_speech():
    """Unload the managed Whisper model."""
    stopped = await asyncio.to_thread(speech.stop)
    return {"status": "ok", "running": False, "stopped": stopped}


def _require_local_microphone_request(request: Request) -> None:
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1"}:
        raise HTTPException(403, "System microphone capture is restricted to localhost.")
    if request.headers.get("x-llamastudio-local") != "speech-capture":
        raise HTTPException(403, "Missing local microphone request header.")


@app.post("/api/speech/recording/start")
async def start_speech_recording(request: Request):
    """Start OS-default microphone capture as a browser fallback."""
    _require_local_microphone_request(request)
    try:
        return await asyncio.to_thread(speech.start_system_recording)
    except SpeechError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/speech/recording/stop")
async def stop_speech_recording(
    request: Request,
    language: str = "auto",
    translate: bool = False,
):
    """Stop OS-default microphone capture and return its transcript."""
    _require_local_microphone_request(request)
    try:
        return await asyncio.to_thread(
            speech.stop_system_recording,
            language=language,
            translate=translate,
        )
    except SpeechError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/speech/transcribe")
async def transcribe_speech(
    request: Request,
    language: str = "auto",
    translate: bool = False,
):
    """Transcribe a raw browser recording through managed whisper.cpp."""
    content_type = request.headers.get("content-type", "").lower()
    if not content_type.startswith("audio/") and content_type != "application/octet-stream":
        raise HTTPException(415, "Upload an audio recording.")

    audio = await request.body()
    if len(audio) > settings.SPEECH_MAX_AUDIO_BYTES:
        raise HTTPException(413, "The recording exceeds the 100 MB limit.")
    try:
        return await asyncio.to_thread(
            speech.transcribe_bytes,
            audio,
            language=language,
            translate=translate,
        )
    except SpeechError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/models/settings")
async def get_all_model_settings():
    """Retrieve settings profiles for all models."""
    return load_model_settings()


@app.post("/api/models/settings")
async def save_one_model_settings(request: Request):
    """Save custom settings profile for a model."""
    body = await request.json()
    model_path = body.get("path")
    model_params = body.get("settings")
    if not model_path:
        raise HTTPException(400, "Model path is required")

    config_loader.save_model_profile(model_path, model_params)
    return {"status": "ok"}


@app.post("/api/models/refresh")
async def refresh_models():
    """Force rescan of model directories."""
    from .model_manager import refresh_models

    models = refresh_models()
    return {
        "models": [
            {
                "path": m.path,
                "name": m.name,
                "size": m.size,
                "size_human": m.size_human,
                "quant": m.quant,
                "is_multimodal": m.is_multimodal,
                "mmproj_path": m.mmproj_path,
                "is_loaded": m.path == server._current_model,
            }
            for m in models
        ]
    }


@app.delete("/api/models/delete")
async def delete_model(request: Request):
    """Delete a GGUF model from the local filesystem."""
    body = await request.json()
    model_path = body.get("path")
    if not model_path:
        raise HTTPException(400, "Model path is required")

    abs_path = Path(model_path).resolve()

    if not abs_path.exists() or not abs_path.is_file():
        raise HTTPException(404, "Model file not found on disk")

    is_safe = False
    for allowed_dir in config_loader.get_model_directories():
        allowed_abs = Path(allowed_dir).resolve()
        if allowed_abs in abs_path.parents:
            is_safe = True
            break

    if not is_safe:
        raise HTTPException(
            403, "Access denied: cannot delete files outside allowed model directories"
        )

    current_model = Path(server._current_model).resolve() if server._current_model else None
    if current_model == abs_path and server.is_running:
        raise HTTPException(
            400, "Cannot delete a model that is currently loaded. Please eject the model first."
        )

    try:
        abs_path.unlink()

        # Clean up empty parent directories up to the allowed MODEL_DIRS
        parent = abs_path.parent
        for allowed_dir in config_loader.get_model_directories():
            allowed_abs = Path(allowed_dir).resolve()
            while parent != allowed_abs and parent.exists() and len(list(parent.iterdir())) == 0:
                parent.rmdir()
                parent = parent.parent

        return {"status": "ok", "message": f"Successfully deleted model {abs_path.name} from disk"}
    except Exception as e:
        logger.error(f"Error deleting model file {abs_path}: {e}")
        raise HTTPException(500, f"Error deleting model: {e}")


# ─── Hugging Face Search & Downloader Endpoints ───────────────


@app.get("/api/models/search")
async def search_models(q: str = "", sort: str = "downloads"):
    """Search Hugging Face GGUF models."""
    from .model_manager import search_huggingface_models

    results = await search_huggingface_models(q, sort)
    return {"models": results}


@app.get("/api/models/hf-details")
async def get_hf_model_details(repo_id: str):
    """Get metadata and README content from a Hugging Face repo."""
    from .model_manager import get_huggingface_model_details, get_huggingface_model_readme

    # Run fetch details and readme concurrently
    details, readme = await asyncio.gather(
        get_huggingface_model_details(repo_id), get_huggingface_model_readme(repo_id)
    )

    if details is None:
        raise HTTPException(404, f"Hugging Face repository '{repo_id}' not found.")

    return {"details": details, "readme": readme}


@app.post("/api/models/download")
async def download_model(request: Request):
    """Trigger background GGUF model download from Hugging Face."""
    body = await request.json()
    repo_id = body.get("repo_id")
    filename = body.get("filename")
    target_model_path = body.get("target_model_path")

    if not repo_id or not filename:
        raise HTTPException(400, "Both repo_id and filename are required.")

    from .downloader import downloader

    if downloader.is_active:
        raise HTTPException(409, "Another download is already in progress.")

    target_dir = None
    if target_model_path:
        model = _get_scanned_model(target_model_path)
        if model is None:
            raise HTTPException(404, "Target model is not present in a configured model directory.")

        from .model_manager import resolve_model_projector

        projector = await resolve_model_projector(model.path)
        if (
            projector.get("status") != "downloadable"
            or projector.get("repo_id") != repo_id
            or projector.get("filename") != filename
        ):
            raise HTTPException(400, "Requested file is not the resolved projector for this model.")
        target_dir = str(Path(model.path).parent)

    success = await downloader.start_download(repo_id, filename, target_dir=target_dir)
    if not success:
        raise HTTPException(500, "Failed to start background download.")

    return {"status": "ok", "message": f"Started download of {filename}"}


@app.get("/api/models/download/progress")
async def get_download_progress():
    """Stream download progress back to frontend in real time via SSE."""
    from .downloader import downloader

    async def progress_generator():
        while True:
            prog = downloader.get_progress()
            yield f"data: {json.dumps(prog)}\n\n"
            if prog.get("status") in ["completed", "failed", "cancelled", "idle"]:
                break
            await asyncio.sleep(1)

    return StreamingResponse(
        progress_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/models/download/cancel")
async def cancel_download():
    """Cancel any active model download task."""
    from .downloader import downloader

    await downloader.cancel_download()
    return {"status": "ok"}


@app.get("/api/models/download/active")
async def is_download_active():
    """Check if a download task is currently active."""
    from .downloader import downloader

    return {"active": downloader.is_active, "progress": downloader.get_progress()}


# ─── Chat ─────────────────────────────────────────────────────


@app.get("/api/chat/conversations")
async def get_conversations():
    return {"conversations": chat.list_conversations()}


@app.post("/api/chat/new")
async def new_conversation():
    conv = chat.new_conversation()
    return {"id": conv.id, "title": conv.title}


@app.post("/api/chat/switch/{conv_id}")
async def switch_conversation(conv_id: str):
    conv = chat.switch_to(conv_id)
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    return {
        "id": conv.id,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "reasoning": m.reasoning,
                "tool_calls": m.tool_calls,
                "tool_call_id": m.tool_call_id,
                "name": m.name,
                "images": [image.to_dict() for image in m.images],
                "audios": [audio.to_dict() for audio in m.audios],
                "vision_recovery": m.vision_recovery,
                "metrics": m.metrics,
            }
            for m in conv.messages
        ],
    }


@app.post("/api/chat/rename/{conv_id}")
async def rename_conversation(conv_id: str, request: Request):
    body = await request.json()
    new_title = body.get("title", "").strip()
    if not new_title:
        raise HTTPException(400, "Title is required")
    if not chat.rename_conversation(conv_id, new_title):
        raise HTTPException(404, "Conversation not found")
    return {"status": "ok", "title": new_title}


@app.delete("/api/chat/{conv_id}")
async def delete_conversation(conv_id: str):
    if not chat.delete_conversation(conv_id):
        raise HTTPException(404, "Conversation not found")
    return {"status": "ok"}


@app.post("/api/chat/send")
async def send_message(request: Request):
    """Receive a message and stream the response back via SSE."""
    body = await request.json()
    user_msg = body.get("message", "").strip()
    try:
        images = validate_image_attachments(body.get("images"))
        audios = validate_audio_attachments(body.get("audios"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    max_tokens = body.get("max_tokens")
    system_prompt = body.get("system_prompt")
    top_k = body.get("top_k")
    min_p = body.get("min_p")
    repeat_penalty = body.get("repeat_penalty")
    stop = body.get("stop")
    enable_thinking = body.get("enable_thinking")
    if enable_thinking is not None and not isinstance(enable_thinking, bool):
        raise HTTPException(400, "enable_thinking must be a boolean")

    if not user_msg and not images and not audios:
        raise HTTPException(400, "Empty message")
    if not user_msg:
        if audios and not images:
            user_msg = "Transcribe this audio and respond to what was said."
        elif images and not audios:
            user_msg = "Describe this image."
        else:
            user_msg = "Analyze the attached media."

    if not server.is_running:
        raise HTTPException(503, "llama-server is not running")
    vision_recovery = None
    if images and not server.supports_multimodal():
        from .model_manager import resolve_model_projector

        vision_recovery = await resolve_model_projector(server._current_model)

    def event_generator():
        yield "data: {'type': 'start'}\n\n"
        yield from chat.stream_chat(
            user_msg,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            top_k=top_k,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            stop=stop,
            images=images,
            audios=audios,
            vision_recovery=vision_recovery,
            enable_thinking=enable_thinking,
        )
        yield "data: {'type': 'end'}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── App lifecycle ────────────────────────────────────────────


@app.on_event("startup")
async def startup():
    """Startup event. The app starts clean without a model loaded."""
    config_loader.ensure_initialized()
    logger.info(
        "[LLamaStudio] Application started. Access interface on "
        f"http://{settings.APP_HOST}:{settings.APP_PORT}"
    )


@app.on_event("shutdown")
async def shutdown():
    """Stop managed inference servers when the app shuts down."""
    logger.info("[LLamaStudio] Shutting down...")
    speech.stop()
    server.eject_model()
