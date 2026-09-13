"""Bounded, filterable backend log tail."""

from __future__ import annotations

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Checkbox, Input, RichLog, Select, Static

from .widgets import StudioView


class LogsView(StudioView):
    def __init__(self):
        super().__init__(id="logs")
        self.loaded = False
        self.last_lines: list[str] = []
        self.last_render: tuple | None = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="panel logs-panel"):
            with Horizontal(classes="toolbar"):
                yield Select(
                    [("llama-server", "llama"), ("Application", "app")],
                    value="llama",
                    allow_blank=False,
                    id="log-source",
                )
                yield Input(placeholder="Filter log lines", id="log-filter")
                yield Checkbox("Follow", True, id="log-follow")
            yield Static(
                "Last 500 lines / refreshes every 3 seconds", id="log-summary", classes="muted"
            )
            yield RichLog(id="log-output", highlight=False, markup=False, wrap=True, max_lines=2000)

    def refresh_view(self) -> None:
        self.run_action(self.fetch_logs, group="logs")

    async def fetch_logs(self) -> None:
        source = str(self.query_one("#log-source", Select).value)
        data = await self.client.get("/api/server/logs", type=source, lines=500)
        if source != str(self.query_one("#log-source", Select).value):
            return
        self.last_lines = data.get("logs", [])
        self.loaded = True
        self.render_log_tail()

    @on(Select.Changed, "#log-source")
    def source_changed(self) -> None:
        self.refresh_view()

    @on(Input.Changed, "#log-filter")
    @on(Checkbox.Changed, "#log-follow")
    def filter_changed(self) -> None:
        self.render_log_tail()

    def render_log_tail(self) -> None:
        query = self.query_one("#log-filter", Input).value.casefold()
        lines = [line.rstrip() for line in self.last_lines if query in line.casefold()]
        following = self.query_one("#log-follow", Checkbox).value
        signature = (tuple(lines), following)
        if signature == self.last_render:
            return
        self.last_render = signature
        output = self.query_one("#log-output", RichLog)
        position = output.scroll_offset
        output.auto_scroll = following
        output.clear()
        for line in lines:
            color = (
                self.app.palette.error
                if "error" in line.casefold()
                else self.app.palette.warning
                if "warn" in line.casefold()
                else self.app.palette.text
            )
            output.write(Text(line, style=color), scroll_end=following)
        if not lines:
            output.write("No matching log lines." if query else "No log entries yet.")
        if not following:
            output.scroll_to(x=position.x, y=position.y, animate=False, force=True)
        self.query_one("#log-summary", Static).update(
            f"{len(lines)} lines / {'Following' if following else 'Scroll position held'} / Ctrl+R Refresh"
        )
