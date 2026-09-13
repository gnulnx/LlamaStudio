"""Responsive terminal branding and truthful, device-wide GPU telemetry."""

from __future__ import annotations

from importlib.metadata import version
from math import isfinite

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

PURPLE = "#b49aff"
GREEN = "#39d99a"
MUTED = "#9390ad"
YELLOW = "#eac86a"

# Plain ASCII: no emoji-width surprises or Nerd Font dependency over SSH.
LLAMA = "  //\n ('>\n / /____\n/      /\n||----||"


def measurement(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value):
        return float(value)
    return None


class StudioHeader(Vertical):
    def compose(self) -> ComposeResult:
        with Horizontal(id="header-full"):
            with Horizontal(id="identity"):
                yield Static(Text(LLAMA, style=PURPLE), id="llama")
                with Vertical(id="wordmark"):
                    yield Static("LlamaStudio TUI", id="brand")
                    yield Static("Local AI. Your terminal.", id="tagline")
            with Horizontal(id="telemetry"):
                with Vertical(id="gpu-stat", classes="header-stat"):
                    yield Static("PRIMARY GPU", classes="stat-label")
                    yield Static("Detecting...", id="gpu-name")
                with Vertical(id="memory-stat", classes="header-stat"):
                    yield Static("VRAM", classes="stat-label", id="memory-label")
                    yield Static("Awaiting telemetry", id="memory-value")
                    yield Static("", id="memory-bar")
                with Vertical(id="server-stat", classes="header-stat"):
                    yield Static("LLAMA-SERVER", classes="stat-label")
                    yield Static("Connecting...", id="connection")
                with Vertical(id="model-stat", classes="header-stat"):
                    yield Static("ACTIVE MODEL", classes="stat-label")
                    yield Static("Connecting...", id="active-model")
            with Vertical(id="header-meta"):
                yield Static(f"v{version('llamastudio')}", id="header-version")
                yield Static("F1 Help", id="header-help")
        yield Static("LlamaStudio TUI / Connecting...", id="header-summary")

    def update_status(self, status: dict, gpu: dict, *, connected: bool = True) -> None:
        name = str(gpu.get("name") or "Unavailable")
        total = measurement(gpu.get("total_vram"))
        used = measurement(gpu.get("used_vram"))
        # Older servers only expose capacity. Never infer usage from model size.
        if "total_vram" not in gpu:
            total = measurement(gpu.get("vram"))
        if total is not None and total <= 0:
            total = None
        if total is None or used is None or not 0 <= used <= total:
            used = None
        unified = gpu.get("memory_kind") == "unified"
        memory_label = "UNIFIED MEMORY" if unified else "VRAM"
        if total is not None and used is not None:
            fraction = used / total
            memory = f"{used:.1f} / {total:.1f} GiB ({fraction:.0%})"
            filled = round(fraction * 24)
            bar = Text("━" * filled, style=PURPLE)
            bar.append("━" * (24 - filled), style="#37304f")
        else:
            memory = f"{total:.1f} GiB / usage N/A" if total is not None else "Unavailable"
            bar = Text("")

        params = status.get("current_params") or {}
        cpu = params.get("cpu_mode") or params.get("gpu_layers") == 0
        mode = "CPU" if cpu else "GPU" if params.get("gpu_layers") is not None else ""
        if not connected:
            state, color, model = "Backend unavailable", YELLOW, "Unavailable"
        elif status.get("is_loading"):
            state, color = "Loading...", YELLOW
            model = str(status.get("current_model_name") or "Loading model...")
        elif status.get("running"):
            state, color = "Running" + (f" / {mode}" if mode else ""), GREEN
            model = str(status.get("current_model_name") or "Loaded model")
        else:
            state, color, model = "Stopped", MUTED, "No model loaded"

        self.query_one("#gpu-name", Static).update(Text(name, style=GREEN if gpu else MUTED))
        self.query_one("#gpu-name").tooltip = f"Primary detected device: {name}"
        self.query_one("#memory-label", Static).update(memory_label)
        self.query_one("#memory-value", Static).update(Text(memory, style=PURPLE))
        self.query_one("#memory-bar", Static).update(bar)
        self.query_one("#memory-stat").tooltip = (
            "Shared system memory capacity; GPU usage is not reported."
            if unified
            else "Device-wide memory usage, including other applications. Refreshed every 3 seconds."
        )
        self.query_one("#connection", Static).update(Text(state, style=color))
        self.query_one("#active-model", Static).update(Text(model, style=PURPLE))
        self.query_one("#active-model").tooltip = model

        # Compact terminals keep the model, device, memory, and server state,
        # without the art or card borders taking away working space.
        summary = Text("LlamaStudio TUI", style=f"bold {PURPLE}")
        summary.append("  /  ", style=MUTED)
        summary.append(model, style=PURPLE)
        summary.truncate(max(1, self.size.width), overflow="ellipsis")
        summary.append("\n")
        short_name = name.removeprefix("NVIDIA GeForce ")
        device = Text(short_name, style=GREEN if gpu else MUTED)
        device.truncate(
            max(8, self.size.width - len(memory) - len(state) - 12), overflow="ellipsis"
        )
        summary.append_text(device)
        summary.append(f"  |  {'RAM' if unified else 'VRAM'} {memory}  |  ", style=MUTED)
        summary.append(state, style=color)
        self.query_one("#header-summary", Static).update(summary)
