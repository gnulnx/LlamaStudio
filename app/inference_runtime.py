"""Discover managed inference without importing application configuration."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def inference_root() -> Path:
    return (
        Path(
            os.environ.get("LLAMASTUDIO_CONFIG_DIR", str(Path.home() / ".config" / "llamastudio"))
        ).expanduser()
        / "inference"
    )


def managed_binary() -> str | None:
    root = inference_root().resolve()
    try:
        manifest = root / "current.json"
        if not manifest.resolve().is_relative_to(root):
            return None
        data = json.loads(manifest.read_text())
        relative = Path(data["binary"])
        binary = (root / relative).resolve()
        if relative.is_absolute() or not binary.is_relative_to(root / "releases"):
            return None
        if binary.is_file():
            return str(binary)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def runtime_environment(binary: str, env: dict[str, str] | None = None) -> dict[str, str]:
    """Limit bundled library lookup to the chosen managed release."""
    result = dict(os.environ if env is None else env)
    base = inference_root().resolve()
    root = (base / "releases").resolve()
    if not root.is_relative_to(base):
        return result
    path = Path(binary).expanduser().resolve()
    if not path.is_relative_to(root):
        return result
    relative = path.relative_to(root)
    if len(relative.parts) < 2:
        return result
    release = root / relative.parts[0]
    directories = {path.parent}
    for pattern in ("*.so*", "*.dylib"):
        for library in release.rglob(pattern):
            if library.resolve().is_relative_to(release):
                directories.add(library.parent)
    variable = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    value = os.pathsep.join(str(directory) for directory in sorted(directories))
    existing = result.get(variable)
    result[variable] = value + (os.pathsep + existing if existing else "")
    return result


def managed_backend(binary: str) -> str | None:
    if managed_binary() != str(Path(binary).resolve()):
        return None
    try:
        data = json.loads((inference_root() / "current.json").read_text())
        return data.get("backend")
    except (OSError, ValueError, TypeError):
        return None
