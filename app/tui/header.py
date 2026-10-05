"""Responsive terminal branding and truthful, device-wide GPU telemetry."""

from __future__ import annotations

from importlib.metadata import version
from math import isfinite

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static


def measurement(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value):
        return float(value)
    return None


class StudioHeader(Vertical):
    def compose(self) -> ComposeResult:
        with Horizontal(id="header-full"):
            with Horizontal(id="identity"), Vertical(id="wordmark"):
                yield Static("LlamaStudio", id="brand")
                yield Static("Local AI. Your Terminal.", id="tagline")
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
        yield Static("LlamaStudio / Connecting...", id="header-summary")

    def update_status(self, status: dict, gpu: dict, *, connected: bool = True) -> None:
        palette = self.app.palette
        params = status.get("current_params") or {}
        devices = gpu.get("devices", [])
        target_device = params.get("gpu_device")
        active_device = None
        if target_device and str(target_device).lower() not in ("all", "auto"):
            dev_str = str(target_device).strip()
            if dev_str.upper().startswith("CUDA"):
                dev_str = dev_str[4:]
            active_device = next(
                (
                    d
                    for d in devices
                    if str(d.get("id")) == dev_str or str(d.get("index")) == dev_str
                ),
                None,
            )

        if active_device:
            name = str(active_device.get("name") or "Unavailable")
            total = measurement(active_device.get("total_vram", active_device.get("vram")))
            used = measurement(active_device.get("used_vram"))
            unified = active_device.get("memory_kind") == "unified"
        elif len(devices) > 1:
            name = f"{len(devices)}x GPUs (Multi-GPU)"
            tot_sys = measurement(gpu.get("total_system_vram"))
            tot_free = measurement(gpu.get("total_system_free"))
            if tot_sys is not None and tot_free is not None:
                total = tot_sys
                used = tot_sys - tot_free
            else:
                total = tot_sys or measurement(gpu.get("total_vram"))
                used = measurement(gpu.get("used_vram"))
            unified = gpu.get("memory_kind") == "unified"
        else:
            name = str(gpu.get("name") or "Unavailable")
            total = measurement(gpu.get("total_vram"))
            used = measurement(gpu.get("used_vram"))
            if "total_vram" not in gpu:
                total = measurement(gpu.get("vram"))
            unified = gpu.get("memory_kind") == "unified"

        if total is not None and total <= 0:
            total = None
        if total is None or used is None or not 0 <= used <= total:
            used = None
        memory_label = "UNIFIED MEMORY" if unified else "VRAM"
        if total is not None and used is not None:
            fraction = used / total
            memory = f"{used:.1f} / {total:.1f} GiB ({fraction:.0%})"
            filled = round(fraction * 24)
            bar = Text("━" * filled, style=palette.warning if fraction >= 0.9 else palette.teal)
            bar.append("━" * (24 - filled), style=palette.border_muted)
        else:
            memory = f"{total:.1f} GiB / usage N/A" if total is not None else "Unavailable"
            bar = Text("")

        cpu = params.get("cpu_mode") or params.get("gpu_layers") == 0
        mode = "CPU" if cpu else "GPU" if params.get("gpu_layers") is not None else ""
        if not connected:
            state, color, model = "Backend unavailable", palette.warning, "Unavailable"
        elif status.get("is_loading"):
            state, color = "Loading...", palette.warning
            model = str(status.get("current_model_name") or "Loading model...")
        elif status.get("running"):
            state, color = "Running" + (f" / {mode}" if mode else ""), palette.success
            model = str(status.get("current_model_name") or "Loaded model")
        else:
            state, color, model = "Stopped", palette.muted, "No model loaded"

        if len(devices) > 1:
            tooltip_parts = [f"Detected {len(devices)} devices:"]
            for d in devices:
                d_id = d.get("id", d.get("index", ""))
                d_n = d.get("name", "GPU")
                d_t = d.get("total_vram", d.get("vram"))
                d_u = d.get("used_vram")
                if d_t is not None and d_u is not None:
                    tooltip_parts.append(f"• GPU {d_id}: {d_n} ({d_u:.1f}/{d_t:.1f} GiB)")
                elif d_t is not None:
                    tooltip_parts.append(f"• GPU {d_id}: {d_n} ({d_t:.1f} GiB)")
                else:
                    tooltip_parts.append(f"• GPU {d_id}: {d_n}")
            gpu_tooltip = "\n".join(tooltip_parts)
        else:
            gpu_tooltip = f"Primary detected device: {name}"

        self.query_one("#gpu-name", Static).update(
            Text(name, style=palette.success if gpu else palette.muted)
        )
        self.query_one("#gpu-name").tooltip = gpu_tooltip
        self.query_one("#memory-label", Static).update(memory_label)
        self.query_one("#memory-value", Static).update(Text(memory, style=palette.teal))
        self.query_one("#memory-bar", Static).update(bar)
        self.query_one("#memory-stat").tooltip = (
            "Shared system memory capacity; GPU usage is not reported."
            if unified
            else "Device-wide memory usage, including other applications. Refreshed every 3 seconds."
        )
        self.query_one("#connection", Static).update(Text(state, style=color))
        self.query_one("#active-model", Static).update(Text(model, style=palette.accent))
        self.query_one("#active-model").tooltip = model

        # Compact terminals keep the model, device, memory, and server state,
        # without card borders taking away working space.
        summary = Text("LlamaStudio", style=f"bold {palette.accent}")
        summary.append("  /  ", style=palette.muted)
        summary.append(model, style=palette.accent)
        summary.truncate(max(1, self.size.width), overflow="ellipsis")
        summary.append("\n")
        short_name = name.removeprefix("NVIDIA GeForce ")
        device = Text(short_name, style=palette.success if gpu else palette.muted)
        device.truncate(
            max(8, self.size.width - len(memory) - len(state) - 12), overflow="ellipsis"
        )
        summary.append_text(device)
        summary.append(f"  |  {'RAM' if unified else 'VRAM'} {memory}  |  ", style=palette.muted)
        summary.append(state, style=color)
        self.query_one("#header-summary", Static).update(summary)
