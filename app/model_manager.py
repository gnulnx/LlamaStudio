"""
Model discovery and management.
Scans GGUF directories and manages model switching.
"""
from __future__ import annotations
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from .config import settings

@dataclass
class ModelInfo:
    path: str
    name: str
    size: int
    size_human: str
    quant: Optional[str] = None
    is_multimodal: bool = False
    is_mmproj: bool = False

def _parse_quant(filename: str) -> Optional[str]:
    """Extract quantization type from filename."""
    quants = ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q3_K_M", "Q3_K_S", "Q2_K", "IQ4_XS", "IQ3_XS"]
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

    for model_dir in settings.MODEL_DIRS:
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
            name = re.sub(r'[-_.]gguf$', '', name, flags=re.IGNORECASE)

            models.append(ModelInfo(
                path=str(gguf),
                name=name,
                size=gguf.stat().st_size,
                size_human=_format_size(gguf.stat().st_size),
                quant=_parse_quant(gguf.name),
                is_multimodal=_is_multimodal(gguf.name),
            ))

    # Sort by size (largest first, usually better models)
    models.sort(key=lambda m: m.size, reverse=True)
    return models

# Cache the model list
_model_cache: Optional[list[ModelInfo]] = None

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
