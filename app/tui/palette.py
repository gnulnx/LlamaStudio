"""One validated palette for Textual CSS and Rich renderables."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields
from importlib.resources import files

from textual.theme import Theme

from app.tools import check_path_safe


@dataclass(frozen=True)
class Palette:
    background: str
    panel: str
    surface: str
    hover: str
    primary: str
    focus: str
    selection: str
    selection_inactive: str
    border: str
    border_muted: str
    text: str
    muted: str
    dim: str
    accent: str
    success: str
    teal: str
    warning: str
    error: str
    on_accent: str

    @classmethod
    def from_mapping(cls, data: dict) -> Palette:
        expected = {field.name for field in fields(cls)}
        if missing := expected - data.keys():
            raise ValueError(f"Missing palette colors: {', '.join(sorted(missing))}")
        if unknown := data.keys() - expected:
            raise ValueError(f"Unknown palette colors: {', '.join(sorted(unknown))}")
        for name, value in data.items():
            if not isinstance(value, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
                raise ValueError(f"Palette color '{name}' must be a #RRGGBB hex color.")
        return cls(**data)

    def theme(self, *, dark: bool = True) -> Theme:
        return Theme(
            name="llamastudio",
            primary=self.primary,
            secondary=self.accent,
            accent=self.focus,
            foreground=self.text,
            background=self.background,
            surface=self.surface,
            panel=self.panel,
            success=self.success,
            warning=self.warning,
            error=self.error,
            dark=dark,
            text_alpha=1.0,
            variables={
                **{f"studio-{key.replace('_', '-')}": value for key, value in asdict(self).items()},
                "border": self.border,
                "border-blurred": self.border_muted,
                "text-muted": self.muted,
                "text-disabled": self.dim,
                "block-cursor-foreground": self.on_accent,
                "block-cursor-background": self.selection,
                "block-cursor-blurred-foreground": self.text,
                "block-cursor-blurred-background": self.selection_inactive,
            },
        )


def _read_colors(source: str) -> dict:
    data = json.loads(source)
    if not isinstance(data, dict):
        raise ValueError("A palette must be a JSON object mapping color roles to #RRGGBB values.")
    return data


def load_palette(path: str | None = None, *, base: Palette | None = None) -> Palette:
    """Load the bundled palette, optionally overridden by a workspace JSON file.

    Revalidate user paths on every reload, including symlink destinations. Bundled
    data is an application resource so installed wheels also work from any cwd.
    """
    colors = (
        asdict(base)
        if base is not None
        else _read_colors(files("app.tui").joinpath("palette.json").read_text(encoding="utf-8"))
    )
    if path is not None:
        colors.update(_read_colors(check_path_safe(path).read_text(encoding="utf-8")))
    return Palette.from_mapping(colors)
