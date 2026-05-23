"""
LLamaStudio - FastAPI backend with HTMX frontend.
A desktop-like chat interface for llama.cpp.
"""
from __future__ import annotations
import os
import json
import shutil
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .config import settings
from .logger import logger
from .server_manager import server
from .chat import chat

app = FastAPI(title="LLamaStudio")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Mount static files
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Helper functions for model settings persistence
def load_model_settings() -> dict:
    path = Path(settings.MODEL_SETTINGS_FILE)
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_model_settings(all_settings: dict):
    path = Path(settings.MODEL_SETTINGS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_settings, f, indent=2)

# ─── Page routes ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    conversations = chat.list_conversations()
    status = server.get_status()
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
            "system_prompt": settings.DEFAULT_SYSTEM_PROMPT,
            "temperature": settings.DEFAULT_TEMPERATURE,
            "top_p": settings.DEFAULT_TOP_P,
            "max_tokens": settings.DEFAULT_MAX_TOKENS,
        }
    )

# ─── Server & Model management ────────────────────────────────

@app.get("/api/server/status")
async def server_status():
    return server.get_status()

@app.get("/api/server/logs")
async def server_logs(type: str = "llama", lines: int = 50):
    log_name = "app.log" if type == "app" else "server.log"
    log_file = Path(settings.LOG_DIR) / log_name
    if not log_file.exists():
        return {"logs": []}
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
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

    result = server.load_model(model_path, model_params)
    if not result:
        raise HTTPException(500, "Failed to load model. Check server logs.")

    return {
        "status": "ok",
        "model": model_path,
        "running": server.is_running
    }

@app.post("/api/models/eject")
async def eject_model():
    """Eject the currently loaded model."""
    server.eject_model()
    return {
        "status": "ok",
        "running": False
    }

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

    all_settings = load_model_settings()
    all_settings[model_path] = model_params
    save_model_settings(all_settings)
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
                "is_loaded": m.path == server._current_model,
            }
            for m in models
        ]
    }



# ─── Hugging Face Search & Downloader Endpoints ────────────────

import asyncio

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
        get_huggingface_model_details(repo_id),
        get_huggingface_model_readme(repo_id)
    )
    
    if details is None:
        raise HTTPException(404, f"Hugging Face repository '{repo_id}' not found.")
        
    return {
        "details": details,
        "readme": readme
    }

@app.post("/api/models/download")
async def download_model(request: Request):
    """Trigger background GGUF model download from Hugging Face."""
    body = await request.json()
    repo_id = body.get("repo_id")
    filename = body.get("filename")
    
    if not repo_id or not filename:
        raise HTTPException(400, "Both repo_id and filename are required.")
        
    from .downloader import downloader
    if downloader.is_active:
        raise HTTPException(409, "Another download is already in progress.")
        
    success = await downloader.start_download(repo_id, filename)
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
        }
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
    return {"id": conv.id, "messages": [
        {
            "role": m.role,
            "content": m.content,
            "timestamp": m.timestamp,
            "tool_calls": m.tool_calls,
            "tool_call_id": m.tool_call_id,
            "name": m.name
        } for m in conv.messages
    ]}

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
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    max_tokens = body.get("max_tokens")
    system_prompt = body.get("system_prompt")
    top_k = body.get("top_k")
    min_p = body.get("min_p")
    repeat_penalty = body.get("repeat_penalty")
    stop = body.get("stop")

    if not user_msg:
        raise HTTPException(400, "Empty message")

    if not server.is_running:
        raise HTTPException(503, "llama-server is not running")

    def event_generator():
        yield "data: {'type': 'start'}\n\n"
        for chunk in chat.stream_chat(
            user_msg,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            top_k=top_k,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            stop=stop,
        ):
            yield chunk
        yield "data: {'type': 'end'}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

# ─── App lifecycle ────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Startup event. The app starts clean without a model loaded."""
    logger.info("[LLamaStudio] Application started. Access interface on http://127.0.0.1:8765")

@app.on_event("shutdown")
async def shutdown():
    """Stop llama-server when app shuts down."""
    logger.info("[LLamaStudio] Shutting down...")
    server.eject_model()

