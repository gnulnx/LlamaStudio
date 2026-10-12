"""Persistent application sandbox settings, separate from generation settings."""

from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request

from .config_store import config_loader
from .sandbox import SandboxPolicy

router = APIRouter()


@router.get("/api/settings/sandbox")
async def get_sandbox_settings():
    return SandboxPolicy.capture().to_dict()


@router.put("/api/settings/sandbox")
async def update_sandbox_settings(request: Request):
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.headers.get("host"):
        raise HTTPException(403, "Settings changes require the same origin.")
    if request.headers.get("x-llamastudio-settings") != "1":
        raise HTTPException(403, "Explicit settings action required.")
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(400, "Settings must be valid JSON.") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "Settings must be an object.")
    root = body.get("root")
    disabled = body.get("disable_sandbox")
    command_policy = body.get("command_policy")
    if not isinstance(root, str) or not root.strip():
        raise HTTPException(400, "Sandbox directory must be a non-empty path.")
    if type(disabled) is not bool or command_policy not in ("allow", "deny"):
        raise HTTPException(400, "Choose valid file restrictions and command permissions.")
    try:
        path = Path(root).expanduser()
        if not path.is_absolute():
            raise ValueError("Use an absolute path on the backend computer.")
        path = path.resolve()
        if not path.is_dir():
            raise ValueError("Sandbox directory must be an existing directory.")
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    config = config_loader.load_app_config()
    config["workspace"].update(
        root=str(path), disable_sandbox=disabled, command_policy=command_policy
    )
    config_loader.save_app_config(config)
    return SandboxPolicy.capture().to_dict()
