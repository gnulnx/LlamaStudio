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

            models.append(
                ModelInfo(
                    path=str(gguf),
                    name=name,
                    size=gguf.stat().st_size,
                    size_human=_format_size(gguf.stat().st_size),
                    quant=_parse_quant(gguf.name),
                    is_multimodal=_is_multimodal(gguf.name),
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
