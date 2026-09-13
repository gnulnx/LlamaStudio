"""Local model library and saved load profiles."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Checkbox, DataTable, Input, Label, Select, Static

from .client import APIError
from .widgets import Confirm, LibraryTable, StudioView


class ModelsView(StudioView):
    def __init__(self):
        super().__init__(id="models")
        self.models: list[dict[str, Any]] = []
        self.profiles: dict[str, Any] = {}
        self.selected_path = ""
        self.loaded = False
        self.busy = False

    def compose(self) -> ComposeResult:
        with Vertical(classes="master panel"):
            with Horizontal(classes="toolbar"):
                yield Input(placeholder="Filter local models", id="local-filter")
                yield Button("Rescan", id="local-refresh")
            yield Static("Your local GGUF library", id="local-summary", classes="muted")
            yield LibraryTable(id="local-table", cursor_type="row", zebra_stripes=True)
            yield Static(
                "Enter Configure   Load and chat using your saved profiles", classes="list-hint"
            )
        with VerticalScroll(classes="detail panel", can_focus=True):
            yield Button("Back to models", id="local-back", classes="back")
            yield Label("Select a model", id="local-title", classes="detail-title")
            yield Static(
                "Select a local GGUF to view its settings.", id="local-info", classes="info-card"
            )
            with Horizontal(classes="actions"):
                yield Button("Load model", id="local-load", variant="primary", disabled=True)
                yield Button("Eject", id="local-eject", disabled=True)
            yield Label("LOAD SETTINGS", classes="eyebrow")
            yield Label("Context length (tokens)")
            yield Input("16384", type="integer", id="model-context")
            yield Label("GPU layers (-1 or 999 = all)")
            yield Input("999", type="integer", id="model-layers")
            yield Label("CPU threads")
            yield Input("4", type="integer", id="model-threads")
            yield Label("KV cache type")
            yield Select(
                [(item, item) for item in ("f16", "q8_0", "q4_0")],
                value="q8_0",
                allow_blank=False,
                id="model-cache",
            )
            yield Checkbox("Flash attention", value=True, id="model-flash")
            yield Checkbox("CPU only", id="model-cpu")
            yield Button("Save profile", id="local-save", disabled=True)
            yield Static(
                "Other saved parameters are preserved. Changes apply on the next load.",
                classes="muted",
            )
            yield Button("Delete model...", id="local-delete", variant="error", disabled=True)

    def refresh_view(self) -> None:
        self.run_action(self.fetch_models, group="refresh")

    async def fetch_models(self, rescan: bool = False) -> None:
        self.query_one("#local-summary", Static).update("Scanning local models...")
        data = await (
            self.client.post("/api/models/refresh") if rescan else self.client.get("/api/models")
        )
        self.models = data.get("models", [])
        self.profiles = await self.client.get("/api/models/settings")
        self.loaded = True
        self.render_table()
        self.query_one("#local-summary", Static).update(
            f"{len(self.models)} local models"
            if self.models
            else "No local models. Download a GGUF in Discover, then rescan."
        )
        if self.selected_path and not any(m["path"] == self.selected_path for m in self.models):
            self.selected_path = ""
        if self.models and not self.selected_path:
            self.select_model(self.models[0]["path"])
        self.update_status()

    @on(Button.Pressed, "#local-refresh")
    def rescan(self) -> None:
        self.run_action(self.fetch_models, True, group="refresh")

    @on(Input.Changed, "#local-filter")
    def filter_changed(self) -> None:
        self.render_table()

    @on(LibraryTable.Resized)
    def render_table(self) -> None:
        table = self.query_one("#local-table", DataTable)
        table.clear(columns=True)
        available = table.size.width or (
            self.app.size.width - 8 if self.app.size.width < 110 else self.app.size.width - 71
        )
        table.add_column("Model", width=max(16, min(64, available - 27)))
        table.add_column("Size", width=10)
        table.add_column("State", width=10)
        query = self.query_one("#local-filter", Input).value.casefold()
        for model in self.models:
            if query in model["name"].casefold():
                table.add_row(
                    Text(model["name"], overflow="ellipsis", no_wrap=True),
                    model["size_human"],
                    "",
                    key=model["path"],
                    height=2 if self.app.size.height >= 35 else 1,
                )
        if self.selected_path in {key.value for key in table.rows}:
            table.move_cursor(row=table.get_row_index(self.selected_path))
        self.update_status()

    def update_status(self) -> None:
        current = self.app.server_status.get("current_model")
        running = self.app.server_status.get("running")
        table = self.query_one("#local-table", DataTable)
        for key in table.rows:
            table.update_cell_at(
                (table.get_row_index(key), 2),
                Text(
                    "Loaded" if key.value == current and running else "Ready",
                    style="#39d99a" if key.value == current and running else "#9693b5",
                ),
            )
        for name in ("load", "save", "delete"):
            self.query_one(f"#local-{name}", Button).disabled = self.busy or not self.selected_path
        self.query_one("#local-eject", Button).disabled = self.busy or not running
        self.query_one("#local-load", Button).label = "Working..." if self.busy else "Load model"
        self.query_one("#local-delete", Button).disabled = (
            self.busy or not self.selected_path or (self.selected_path == current and bool(running))
        )

    @on(DataTable.RowHighlighted, "#local-table")
    def highlighted(self, event: DataTable.RowHighlighted) -> None:
        table = self.query_one("#local-table", DataTable)
        if not table.row_count or table.ordered_rows[table.cursor_row].key != event.row_key:
            return
        self.select_model(str(event.row_key.value))

    @on(DataTable.RowSelected, "#local-table")
    def selected(self, event: DataTable.RowSelected) -> None:
        self.select_model(str(event.row_key.value))
        self.show_details()

    def select_model(self, path: str) -> None:
        if path == self.selected_path:
            return
        model = next((model for model in self.models if model["path"] == path), None)
        if not model:
            return
        self.selected_path = path
        profile = self.profiles.get(path) or {}
        self.query_one("#local-title", Label).update(Text(model["name"]))
        self.query_one("#local-info", Static).update(
            Text(f"{model['size_human']} / {model.get('quant', 'GGUF')}\n{path}")
        )
        for widget, key, default in (
            ("context", "ctx_size", 16384),
            ("layers", "gpu_layers", 999),
            ("threads", "threads", 4),
        ):
            self.query_one(f"#model-{widget}", Input).value = str(profile.get(key, default))
        cache = str(profile.get("kv_cache_type", "q8_0"))
        cache_select = self.query_one("#model-cache", Select)
        cache_select.set_options([
            (item, item) for item in dict.fromkeys(("f16", "q8_0", "q4_0", cache))
        ])
        cache_select.value = cache
        self.query_one("#model-flash", Checkbox).value = bool(profile.get("flash_attn", True))
        self.query_one("#model-cpu", Checkbox).value = bool(profile.get("cpu_mode", False))
        self.update_status()

    def edited_profile(self) -> dict[str, Any]:
        profile = deepcopy(self.profiles.get(self.selected_path) or {})
        for widget, key, minimum in (
            ("context", "ctx_size", 128),
            ("layers", "gpu_layers", -1),
            ("threads", "threads", 1),
        ):
            try:
                value = int(self.query_one(f"#model-{widget}", Input).value)
            except ValueError as exc:
                raise APIError(f"{key} must be an integer.") from exc
            if value < minimum:
                raise APIError(f"{key} must be at least {minimum}.")
            profile[key] = value
        profile.update(
            kv_cache_type=str(self.query_one("#model-cache", Select).value),
            flash_attn=self.query_one("#model-flash", Checkbox).value,
            cpu_mode=self.query_one("#model-cpu", Checkbox).value,
        )
        return profile

    @on(Button.Pressed, "#local-load")
    @on(Button.Pressed, "#local-save")
    def settings_action(self, event: Button.Pressed) -> None:
        try:
            profile = self.edited_profile()
        except APIError as exc:
            self.app.notify(str(exc), severity="error")
            return
        path = self.selected_path
        load = event.button.id == "local-load"
        if load and self.app.server_status.get("running"):
            current = self.app.server_status.get("current_model_name", "the active model")
            self.app.push_screen(
                Confirm(
                    "Load model",
                    f"This will unload {current} and load your selection with these settings.",
                    "Load",
                ),
                lambda yes: (
                    self.run_action(self.save_or_load, path, profile, True) if yes else None
                ),
            )
        else:
            self.run_action(self.save_or_load, path, profile, load)

    async def save_or_load(self, path: str, profile: dict[str, Any], load: bool) -> None:
        self.busy = True
        self.update_status()
        try:
            await self.client.post(
                "/api/models/load" if load else "/api/models/settings", path=path, settings=profile
            )
            self.profiles[path] = profile
            self.app.notify("Model loaded. Open Chat to begin." if load else "Model profile saved.")
            await self.app.poll_status()
        finally:
            self.busy = False
            self.update_status()

    @on(Button.Pressed, "#local-eject")
    def request_eject(self) -> None:
        self.app.push_screen(
            Confirm("Eject model", "Unload the active model and free its memory?", "Eject"),
            lambda yes: self.run_action(self.eject) if yes else None,
        )

    async def eject(self) -> None:
        await self.client.post("/api/models/eject")
        await self.app.poll_status()
        self.app.notify("Model ejected.")

    @on(Button.Pressed, "#local-delete")
    def request_delete(self) -> None:
        path = self.selected_path
        self.app.push_screen(
            Confirm(
                "Delete model from disk",
                f"Permanently delete this GGUF?\n\n{path}\n\nIt will need to be downloaded again.",
                "Delete",
            ),
            lambda yes: self.run_action(self.delete, path) if yes else None,
        )

    async def delete(self, path: str) -> None:
        await self.client.request("DELETE", "/api/models/delete", json={"path": path})
        self.app.notify("GGUF deleted from disk. It can be downloaded again from the Hub.")
        await self.fetch_models(True)

    @on(Button.Pressed, "#local-back")
    def back(self) -> None:
        self.show_details(False)
