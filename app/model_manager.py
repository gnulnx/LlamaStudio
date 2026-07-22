"""
Model discovery and management.
Scans GGUF directories and manages model switching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config_store import config_loader


@dataclass
class ModelInfo:
    path: str
    name: str
    size: int
    size_human: str
    quant: str | None = None
    is_multimodal: bool = False
    is_mmproj: bool = False
    mmproj_path: str | None = None


def _parse_quant(filename: str) -> str | None:
    """Extract quantization type from filename."""
    quants = [
        "Q8_0",
        "Q6_K",
        "Q5_K_M",
        "Q5_K_S",
        "Q4_K_M",
        "Q4_K_S",
        "Q3_K_M",
        "Q3_K_S",
        "Q2_K",
        "IQ4_XS",
        "IQ3_XS",
    ]
    for q in quants:
        if q in filename.upper():
            return q
    return None


def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _is_mmproj(filename: str) -> bool:
    """Check if this is a multimodal projector file."""
    return "mmproj" in filename.lower()


def _is_multimodal(filename: str) -> bool:
    """Check if this is a multimodal model (has mmproj in directory)."""
    return "mmproj" in filename.lower() or "vision" in filename.lower()


def _projector_rank(path: str | Path) -> tuple[int, str]:
    name = Path(path).name.lower()
    if "f32" in name:
        quality_rank = 0
    elif "bf16" in name or "f16" in name:
        quality_rank = 1
    elif "q8" in name:
        quality_rank = 2
    elif "q6" in name:
        quality_rank = 3
    elif "q5" in name:
        quality_rank = 4
    elif "q4" in name:
        quality_rank = 5
    else:
        quality_rank = 6
    return quality_rank, name


def find_mmproj(model_path: str | Path) -> str | None:
    """Find the preferred multimodal projector beside a GGUF model."""
    model_file = Path(model_path)
    if not model_file.parent.exists():
        return None

    candidates = [
        path
        for path in model_file.parent.glob("*.gguf")
        if path.is_file() and _is_mmproj(path.name)
    ]
    if not candidates:
        return None

    return str(min(candidates, key=_projector_rank))


def infer_huggingface_repo_id(model_path: str | Path) -> str | None:
    """Infer the Hub repo from the standard <root>/<author>/<repo>/<file> layout."""
    model_file = Path(model_path).expanduser().resolve()
    for model_dir in config_loader.get_model_directories():
        try:
            relative = model_file.relative_to(Path(model_dir).expanduser().resolve())
        except ValueError:
            continue
        if len(relative.parts) >= 3:
            return f"{relative.parts[0]}/{relative.parts[1]}"
    return None


async def resolve_model_projector(model_path: str | Path) -> dict[str, Any]:
    """Resolve a local or downloadable projector for a scanned model."""
    model_file = Path(model_path).expanduser().resolve()
    local_projector = find_mmproj(model_file)
    if local_projector:
        return {
            "status": "local",
            "model_path": str(model_file),
            "projector_path": local_projector,
        }

    repo_id = infer_huggingface_repo_id(model_file)
    if not repo_id:
        return {"status": "unavailable", "model_path": str(model_file)}

    details = await get_huggingface_model_details(repo_id)
    if not details:
        return {
            "status": "unavailable",
            "model_path": str(model_file),
            "repo_id": repo_id,
        }

    candidates = [
        sibling
        for sibling in details.get("siblings", [])
        if isinstance(sibling.get("rfilename"), str)
        and sibling["rfilename"].lower().endswith(".gguf")
        and _is_mmproj(Path(sibling["rfilename"]).name)
    ]
    if not candidates:
        return {
            "status": "unavailable",
            "model_path": str(model_file),
            "repo_id": repo_id,
        }

    selected = min(candidates, key=lambda item: _projector_rank(item["rfilename"]))
    return {
        "status": "downloadable",
        "model_path": str(model_file),
        "repo_id": repo_id,
        "filename": selected["rfilename"],
        "size": selected.get("size"),
    }


def scan_models() -> list[ModelInfo]:
    """Scan model directories for GGUF files and return model info."""
    models = []
    seen_paths = set()

    for model_dir in config_loader.get_model_directories():
        base = Path(model_dir)
        if not base.exists():
            continue

        for gguf in base.rglob("*.gguf"):
            if gguf in seen_paths:
                continue
            seen_paths.add(gguf)

            # Skip projector files (they're not standalone models)
            if _is_mmproj(gguf.name):
                continue

            name = gguf.stem
            # Clean up the name by removing common prefixes/suffixes
            name = re.sub(r"[-_.]gguf$", "", name, flags=re.IGNORECASE)
            mmproj_path = find_mmproj(gguf)

            models.append(
                ModelInfo(
                    path=str(gguf),
                    name=name,
                    size=gguf.stat().st_size,
                    size_human=_format_size(gguf.stat().st_size),
                    quant=_parse_quant(gguf.name),
                    is_multimodal=bool(mmproj_path) or _is_multimodal(gguf.name),
                    mmproj_path=mmproj_path,
                )
            )

    # Sort by size (largest first, usually better models)
    models.sort(key=lambda m: m.size, reverse=True)
    return models


# Cache the model list
_model_cache: list[ModelInfo] | None = None


def get_models() -> list[ModelInfo]:
    """Get model list, scanning if cache is stale."""
    global _model_cache
    if _model_cache is None:
        _model_cache = scan_models()
    return _model_cache


def refresh_models() -> list[ModelInfo]:
    """Force rescan and return updated model list."""
    global _model_cache
    _model_cache = scan_models()
    return _model_cache


async def search_huggingface_models(query: str, sort: str = "downloads") -> list[dict[str, Any]]:
    """Search Hugging Face models by query with a GGUF filter."""
    url = "https://huggingface.co/api/models"
    params = {"search": query, "filter": "gguf", "sort": sort, "limit": 30, "full": "true"}

    headers = {"User-Agent": "LLamaStudio-Client"}

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code != 200:
                return []
            return resp.json()
        except Exception as e:
            from .logger import logger

            logger.error(f"[model_manager] Error searching HF models: {e}")
            return []


async def get_huggingface_model_details(repo_id: str) -> dict[str, Any] | None:
    """Fetch complete metadata of a Hugging Face repository including file sizes."""
    url = f"https://huggingface.co/api/models/{repo_id}"
    headers = {"User-Agent": "LLamaStudio-Client"}

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception as e:
            from .logger import logger

            logger.error(f"[model_manager] Error fetching HF details: {e}")
            return None


async def get_huggingface_model_readme(repo_id: str) -> str:
    """Download the raw README markdown of a model from Hugging Face."""
    url = f"https://huggingface.co/{repo_id}/raw/main/README.md"
    headers = {"User-Agent": "LLamaStudio-Client"}

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                # Try lowercase readme.md fallback
                url_fallback = f"https://huggingface.co/{repo_id}/raw/main/readme.md"
                resp = await client.get(url_fallback, headers=headers)
                if resp.status_code != 200:
                    return f"# {repo_id}\nNo README.md found in this repository."
            return resp.text
        except Exception as e:
            from .logger import logger

            logger.error(f"[model_manager] Error fetching HF README: {e}")
            return f"# {repo_id}\nError retrieving repository documentation: {e}"
