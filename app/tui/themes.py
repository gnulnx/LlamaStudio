"""Theme sources share one contract; the terminal shell has no platform detection.

Adapters resolve immutable snapshots and may request background polling. A platform
adapter owns detection, bounded I/O, and caching (e.g. by file revision). It never
touches widgets. Register its detection factory in SYSTEM_ADAPTERS.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.resources import files

from .palette import Palette, _read_colors, load_palette

THEME_NAMES = ("default", "light", "dark", "system")


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
        if name not in THEME_NAMES[:3]:
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


# Ordered, explicit extension point. A factory returns None if unsupported.
# Only System invokes these factories; there are no platform adapters yet.
SYSTEM_ADAPTERS: tuple[Callable[[], ThemeAdapter | None], ...] = ()


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
            "System theme unavailable; using Default. No system adapter is installed.",
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
