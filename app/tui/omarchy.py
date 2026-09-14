"""Follow the active Omarchy desktop theme. Does nothing on systems without Omarchy."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

from .palette import Palette, load_palette

# Omarchy rewrites this file (new inode and mtime) on every theme change.
THEME_COLORS = Path.home() / ".local" / "state" / "omarchy" / "current" / "theme" / "colors.toml"


def omarchy_detected() -> bool:
    return THEME_COLORS.is_file() and shutil.which("omarchy-theme-color") is not None


def theme_signature() -> tuple[int, int] | None:
    try:
        stat = THEME_COLORS.stat()
    except OSError:
        return None
    return stat.st_ino, stat.st_mtime_ns


def _mix(start: str, end: str, amount: float) -> str:
    pairs = ((int(start[i : i + 2], 16), int(end[i : i + 2], 16)) for i in (1, 3, 5))
    return "#" + "".join(f"{round(a + (b - a) * amount):02x}" for a, b in pairs)


def omarchy_palette() -> tuple[Palette, bool]:
    """Return the active theme as a palette (status colors stay bundled) and whether it is dark."""
    try:
        output = subprocess.run(
            ["omarchy-theme-color", "--file", str(THEME_COLORS), "--all"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        theme = dict(line.split("\t", 1) for line in output.splitlines() if "\t" in line)
        background, foreground, accent = theme["background"], theme["foreground"], theme["accent"]
    except (OSError, KeyError, subprocess.SubprocessError) as exc:
        raise ValueError(f"Could not read the Omarchy theme: {exc}") from exc
    roles = {
        "background": background,
        "panel": _mix(background, foreground, 0.05),
        "surface": _mix(background, foreground, 0.09),
        "hover": _mix(background, accent, 0.20),
        "primary": accent,
        "focus": _mix(accent, foreground, 0.20),
        "selection": theme.get("selection_background", accent),
        "selection_inactive": _mix(background, accent, 0.25),
        "border": accent,
        "border_muted": _mix(background, foreground, 0.18),
        "text": foreground,
        "muted": _mix(foreground, background, 0.34),
        "dim": _mix(foreground, background, 0.52),
        "accent": _mix(accent, foreground, 0.30),
        "on_accent": theme.get("selection_foreground", foreground),
    }
    return Palette.from_mapping({**asdict(load_palette()), **roles}), theme.get("mode") != "light"
