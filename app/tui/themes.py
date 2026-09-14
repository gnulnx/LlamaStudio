"""Theme sources share one contract; the terminal shell has no platform detection.

Adapters resolve immutable snapshots and may request background polling. A platform
adapter owns detection, bounded I/O, and caching (e.g. by file revision). It never
touches widgets. Register its detection factory in SYSTEM_ADAPTERS.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.resources import files

from .palette import Palette, _read_colors, load_palette

BUNDLED_THEME_NAMES = ("default", "light", "dark", "slate")
THEME_NAMES = (*BUNDLED_THEME_NAMES, "system")


@dataclass(frozen=True)
class ResolvedTheme:
    palette: Palette
    dark: bool
    source: str
    notice: str = ""

    def __post_init__(self) -> None:
        Palette.from_mapping(asdict(self.palette))
        if not isinstance(self.dark, bool):
            raise ValueError("Theme dark mode must be a boolean.")


class ThemeAdapter(ABC):
    # None means static: no watcher or idle work. Resolution is serialized and
    # runs off the UI thread. Polling adapters should cache unchanged snapshots.
    refresh_interval: float | None = None

    @abstractmethod
    def resolve(self) -> ResolvedTheme:
        """Return current validated colors, or raise OSError/ValueError on failure."""


class BundledThemeAdapter(ThemeAdapter):
    def __init__(self, name: str = "default"):
        if name not in BUNDLED_THEME_NAMES:
            raise ValueError(f"Unknown bundled theme: {name}")
        self.name = name

    def resolve(self) -> ResolvedTheme:
        if self.name == "default":
            palette = load_palette()
        else:
            source = files("app.tui").joinpath(f"{self.name}.json").read_text(encoding="utf-8")
            palette = Palette.from_mapping(_read_colors(source))
        return ResolvedTheme(palette, self.name != "light", self.name)


class CustomThemeAdapter(ThemeAdapter):
    def __init__(self, path: str, base: str = "default"):
        self.path = path
        self.base = BundledThemeAdapter(base)

    def resolve(self) -> ResolvedTheme:
        base = self.base.resolve()
        return ResolvedTheme(load_palette(self.path, base=base.palette), base.dark, "custom")


def _read_setting(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Only read native appearance preferences, without shell or automation access."""
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Could not read system appearance preferences.") from exc


class AppearanceThemeAdapter(ThemeAdapter):
    """Map native light/dark preference to bundled colors, caching each mode."""

    refresh_interval = 2.0
    source: str

    def __init__(self):
        self._resolved: ResolvedTheme | None = None

    @abstractmethod
    def is_dark(self) -> bool:
        """Read the effective native appearance; failures must not imply light."""

    def resolve(self) -> ResolvedTheme:
        dark = self.is_dark()
        if self._resolved is None or dark != self._resolved.dark:
            palette = BundledThemeAdapter("dark" if dark else "light").resolve().palette
            self._resolved = ResolvedTheme(palette, dark, self.source)
        return self._resolved


class GtkThemeAdapter(AppearanceThemeAdapter):
    source = "gnome-gtk"

    def __init__(self, command: str):
        super().__init__()
        self.command = command

    def is_dark(self) -> bool:
        prefix = [self.command, "get", "org.gnome.desktop.interface"]
        result = _read_setting([*prefix, "color-scheme"])
        scheme = result.stdout.strip().strip("'\"")
        if result.returncode == 0:
            if scheme in ("prefer-dark", "prefer-light"):
                return scheme == "prefer-dark"
            if scheme != "default":
                raise ValueError("Unrecognized GNOME color-scheme preference.")
        elif "No such key" not in result.stderr:
            raise ValueError("Could not read GNOME color-scheme preference.")
        # Older GNOME and Pop!_OS themes may express appearance only through
        # gtk-theme. Explicit modern light/dark preference always wins.
        result = _read_setting([*prefix, "gtk-theme"])
        theme = result.stdout.strip().strip("'\"")
        if result.returncode != 0 or not theme:
            raise ValueError("Could not read GTK theme preference.")
        return "dark" in re.split(r"[-_\s:]", theme.lower())


class MacOSThemeAdapter(AppearanceThemeAdapter):
    source = "macos"

    def is_dark(self) -> bool:
        result = _read_setting(["/usr/bin/defaults", "read", "-g", "AppleInterfaceStyle"])
        if result.returncode == 0:
            value = result.stdout.strip().lower()
            if value in ("dark", "light"):
                return value == "dark"
        # macOS normally removes this key in light mode. Do not turn command
        # failures/timeouts into a light-mode switch. No System Events permission.
        elif (
            result.returncode == 1
            and "AppleInterfaceStyle" in result.stderr
            and "does not exist" in result.stderr
        ):
            return False
        raise ValueError("Could not read macOS appearance preference.")


def detect_macos_theme() -> ThemeAdapter | None:
    return MacOSThemeAdapter() if sys.platform == "darwin" else None


def detect_gtk_theme() -> ThemeAdapter | None:
    desktops = set(os.environ.get("XDG_CURRENT_DESKTOP", "").lower().split(":"))
    if sys.platform.startswith("linux") and desktops & {
        "gnome",
        "pop",
        "cinnamon",
        "budgie",
        "unity",
        "pantheon",
    }:
        command = shutil.which("gsettings")
        if command:
            return GtkThemeAdapter(command)
    return None


# Ordered, explicit extension point. Specific full-palette integrations can be
# registered before these native light/dark adapters. Only System detects them.
SYSTEM_ADAPTERS: tuple[Callable[[], ThemeAdapter | None], ...] = (
    detect_macos_theme,
    detect_gtk_theme,
)


class SystemThemeAdapter(ThemeAdapter):
    def __init__(self):
        self.adapter: ThemeAdapter | None = None
        self.detected = False

    @property
    def refresh_interval(self) -> float | None:
        return self.adapter.refresh_interval if self.adapter else None

    def resolve(self) -> ResolvedTheme:
        if not self.detected:
            for detect in SYSTEM_ADAPTERS:
                self.adapter = detect()
                if self.adapter is not None:
                    break
            self.detected = True
        if self.adapter is not None:
            return self.adapter.resolve()
        return ResolvedTheme(
            load_palette(),
            True,
            "default",
            "System theme unavailable; using Default. No supported desktop was detected.",
        )


def create_theme_adapter(name: str = "default", palette_path: str | None = None) -> ThemeAdapter:
    if name not in THEME_NAMES:
        raise ValueError(f"Unknown theme: {name}")
    # An explicit palette bypasses system detection. Light/Dark select its base.
    if palette_path is not None:
        return CustomThemeAdapter(palette_path, "default" if name == "system" else name)
    return SystemThemeAdapter() if name == "system" else BundledThemeAdapter(name)


def resolve_initial_theme(adapter: ThemeAdapter) -> ResolvedTheme:
    try:
        return adapter.resolve()
    except (OSError, ValueError) as exc:
        if not isinstance(adapter, SystemThemeAdapter):
            raise
        return ResolvedTheme(load_palette(), True, "default", f"System theme unavailable: {exc}")
