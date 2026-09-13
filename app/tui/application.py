"""LlamaStudio terminal shell. Network work never blocks terminal input."""

from __future__ import annotations

import asyncio
from typing import ClassVar
from urllib.parse import urljoin, urlparse

from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.theme import Theme
from textual.widgets import Button, ContentSwitcher, Footer, Markdown, ProgressBar, Static

from .chat import ChatView
from .client import APIError, StudioClient
from .discover import DiscoverView
from .header import StudioHeader
from .logs import LogsView
from .models import ModelsView
from .widgets import Confirm, Help, StudioView


class StudioApp(App[None]):
    TITLE = "LlamaStudio TUI"
    CSS_PATH = "studio.tcss"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS: ClassVar = [
        Binding("f1", "help", "Help"),
        Binding("f2", "view('discover')", "Discover", show=False),
        Binding("f3", "view('models')", "Models", show=False),
        Binding("f4", "view('chat')", "Chat", show=False),
        Binding("f5", "view('logs')", "Logs", show=False),
        Binding("ctrl+r", "refresh", "Refresh", priority=True),
        Binding("ctrl+n", "new_chat", "New chat", show=False),
        Binding("escape", "back", "Back", show=False),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(
        self, base_url: str, *, initial_view: str = "discover", client: StudioClient | None = None
    ):
        super().__init__()
        self.client = client or StudioClient(base_url)
        self.base_url = base_url
        self.current_view = initial_view
        self.server_status: dict = {}
        self.gpu: dict = {}
        self.download_active = False
        self._polling = False
        self.connected = False
        self._last_download_state = "idle"
        self._views: dict[str, StudioView] = {}
        self.register_theme(
            Theme(
                name="llamastudio",
                primary="#8b5cf6",
                secondary="#b49aff",
                accent="#a78bfa",
                foreground="#c9c5e4",
                background="#101019",
                surface="#151521",
                panel="#191825",
                success="#39d99a",
                warning="#eac86a",
                error="#f28b9b",
                dark=True,
            )
        )
        self.theme = "llamastudio"

    def compose(self) -> ComposeResult:
        yield StudioHeader(id="masthead")
        with Horizontal(id="workspace"):
            with Vertical(id="navigation"):
                yield Button("F2 Discover", id="nav-discover", classes="nav-button")
                yield Button("F3 Models", id="nav-models", classes="nav-button")
                yield Button("F4 Chat", id="nav-chat", classes="nav-button")
                yield Button("F5 Logs", id="nav-logs", classes="nav-button")
            with ContentSwitcher(initial=self.current_view, id="sections"):
                yield DiscoverView()
                yield ModelsView()
                yield ChatView()
                yield LogsView()
        with Horizontal(id="download-tray"):
            yield Static("", id="download-label")
            yield ProgressBar(total=100, show_eta=False, id="download-progress")
            yield Button("Cancel", id="download-cancel")
        yield Footer()

    def on_mount(self) -> None:
        self._views = {
            name: self.query_one(f"#{name}", StudioView)
            for name in ("discover", "models", "chat", "logs")
        }
        self.query_one("#download-tray").display = False
        self.apply_layout()
        self.action_view(self.current_view)
        self.tick()
        self.set_interval(3, self.tick)

    def tick(self) -> None:
        if not self.is_running:
            return
        if not self._polling:
            self.run_worker(self.poll_status, group="status")
        if self.current_view == "logs":
            self._views["logs"].refresh_view()

    async def poll_status(self) -> None:
        if self._polling or not self.is_running:
            return
        self._polling = True
        try:
            try:
                self.server_status = await self.client.get("/api/server/status")
                self.connected = True
            except APIError:
                self.connected = False
                self.server_status = {}
                self.gpu = {}
                return
            try:
                self.gpu = await self.client.get("/api/gpu")
            except APIError:
                self.gpu = {}
            data = await self.client.get("/api/models/download/active")
            # HTTP completion may race application shutdown. From here to the
            # end of the render update there are no awaits, so one guard covers
            # all widget access (not just the header in the finally block).
            if not self.is_running:
                return
            self.download_active = bool(data.get("active"))
            progress = data.get("progress") or {}
            state = progress.get("status", "idle")
            self.query_one("#download-tray").display = self.download_active
            if self.download_active:
                self.query_one("#download-label", Static).update(
                    Text(
                        f"{progress.get('filename', 'Downloading...')} / {progress.get('speed_mb', 0):.1f} MiB/s"
                    )
                )
                self.query_one("#download-progress", ProgressBar).update(
                    total=100 if progress.get("total_bytes") else None,
                    progress=progress.get("percent", 0),
                )
            if state != self._last_download_state and self._last_download_state == "downloading":
                if state == "completed":
                    self.notify("Download complete. Your model is available in Models.")
                    self._views["models"].refresh_view()
                elif state == "failed":
                    self.notify(
                        str(progress.get("error") or "Download failed."),
                        severity="error",
                        timeout=10,
                    )
                elif state == "cancelled":
                    self.notify("Download cancelled. Partial data is retained for resume.")
            self._last_download_state = state
        except APIError:
            # An unavailable downloads endpoint is not an inference-server outage.
            pass
        finally:
            self._polling = False
            self.update_header()
            if self.is_running and self._views:
                self._views["discover"].update_memory()
                self._views["models"].update_status()

    def on_resize(self, event: events.Resize) -> None:
        if self._views:
            self.apply_layout(event.size.width, event.size.height)
            self.call_after_refresh(self._views["discover"].render_table)
            self.call_after_refresh(self._views["models"].render_table)

    def apply_layout(self, width: int | None = None, height: int | None = None) -> None:
        # A horizontal navigation bar replaces the rail below 110 columns.
        compact = (width if width is not None else self.size.width) < 110
        self.screen_stack[0].set_class(compact, "compact")
        self.screen_stack[0].set_class(
            (height if height is not None else self.size.height) < 30, "short"
        )
        self.query_one(StudioHeader).set_class(
            (width if width is not None else self.size.width) < 150
            or (height if height is not None else self.size.height) < 30,
            "condensed",
        )
        self.call_after_refresh(self.update_header)
        for number, name in enumerate(self._views, 2):
            self.query_one(
                f"#nav-{name}", Button
            ).label = f"F{number}{' ' if compact else chr(10)}{name.title()}"

    def update_header(self) -> None:
        # A cancelled polling worker can finish while child widgets unmount.
        if not self.is_running:
            return
        self.query_one(StudioHeader).update_status(
            self.server_status, self.gpu, connected=self.connected
        )

    @on(Button.Pressed, ".nav-button")
    def navigate(self, event: Button.Pressed) -> None:
        self.action_view(event.button.id.removeprefix("nav-"))

    def action_view(self, view: str) -> None:
        if len(self.screen_stack) > 1:
            return
        self.current_view = view
        self.query_one("#sections", ContentSwitcher).current = view
        for name in self._views:
            self.query_one(f"#nav-{name}", Button).set_class(name == view, "active")
        section = self._views[view]
        if not section.loaded:
            section.refresh_view()
        target = {
            "discover": "#hub-table",
            "models": "#local-table",
            "chat": "#chat-input"
            if section.has_class("show-detail") or self.size.width >= 110
            else "#chat-conversations",
            "logs": "#log-output",
        }[view]
        self.call_after_refresh(self.query_one(target).focus)

    def action_refresh(self) -> None:
        if len(self.screen_stack) == 1:
            self._views[self.current_view].refresh_view()
            self.tick()

    def action_back(self) -> None:
        if self.size.width < 110:
            self._views[self.current_view].show_details(False)

    def action_new_chat(self) -> None:
        if self.current_view == "chat":
            self._views["chat"].new_requested()

    def action_help(self) -> None:
        self.push_screen(Help())

    @on(Markdown.LinkClicked)
    def open_markdown_link(self, event: Markdown.LinkClicked) -> None:
        event.stop()
        if event.href.startswith("#"):
            event.markdown.goto_anchor(event.href[1:])
            return
        url = event.href
        if event.markdown.id == "hub-readme":
            repo = self._views["discover"].repo_id
            url = urljoin(f"https://huggingface.co/{repo}/blob/main/", url)
        if urlparse(url).scheme in {"https", "http"}:
            self.open_url(url)

    def action_quit(self) -> None:
        if self._views.get("chat") and self._views["chat"].streaming:
            self.push_screen(
                Confirm(
                    "Close the TUI?",
                    "A response is still streaming. Closing disconnects this view; the backend may finish its current operation. Models and downloads remain running.",
                    "Close TUI",
                ),
                lambda yes: self.exit() if yes else None,
            )
        else:
            self.exit()

    @on(Button.Pressed, "#download-cancel")
    def cancel_download(self) -> None:
        self.push_screen(
            Confirm(
                "Cancel download",
                "Stop the current download? Partial data is retained for resume.",
                "Stop download",
            ),
            lambda yes: self._views["discover"].run_action(self.stop_download) if yes else None,
        )

    async def stop_download(self) -> None:
        await self.client.post("/api/models/download/cancel")
        await self.poll_status()

    async def on_unmount(self) -> None:
        await self.client.close()

    async def capture(self, path: str, size: tuple[int, int]) -> None:
        """Render the real backend into an SVG for a reproducible visual review."""
        async with self.run_test(size=size) as pilot:
            # Details workers may be created by the search worker after it returns.
            for _ in range(3):
                await asyncio.wait_for(self.workers.wait_for_complete(), timeout=60)
                await pilot.pause()
            self.save_screenshot(filename=path)
