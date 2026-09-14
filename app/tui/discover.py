"""Hugging Face discovery, file inspection, and shared background downloads."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    Link,
    Markdown,
    ProgressBar,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from .palette import Palette
from .widgets import Confirm, LibraryTable, StudioView, size_label


def capabilities(model: dict[str, Any], palette: Palette) -> Text:
    tags = set(model.get("tags") or [])
    pipeline = model.get("pipeline_tag") or ""
    badges = Text(no_wrap=True, overflow="ellipsis")
    for enabled, label, color in (
        (bool(tags & {"vision", "multimodal"}) or "image" in pipeline, "Vision", palette.warning),
        ("reasoning" in tags, "Reasoning", palette.teal),
        (bool(tags & {"tools", "tool-use", "function-calling"}), "Tools", palette.warning),
    ):
        if enabled:
            if badges:
                badges.append(" ")
            badges.append(f" {label} ", style=f"bold {color} on {palette.surface}")
    return badges if badges else Text(" Text ", style=f"{palette.muted} on {palette.surface}")


class DiscoverView(StudioView):
    def __init__(self):
        super().__init__(id="discover")
        self.models: list[dict[str, Any]] = []
        self.repo_id = ""
        self.details: dict[str, Any] = {}
        self.files: dict[str, dict[str, Any]] = {}
        self.loaded = False
        self.downloading = False

    def compose(self) -> ComposeResult:
        with Vertical(classes="master panel"):
            with Horizontal(classes="toolbar discover-toolbar"):
                yield Input(placeholder="Search Hugging Face / GGUF models", id="hub-query")
                yield Select(
                    [("Most likes", "likes"), ("Downloads", "downloads"), ("Updated", "modified")],
                    value="likes",
                    allow_blank=False,
                    id="hub-sort",
                )
                yield Button("Search", id="hub-search", variant="primary")
            yield Static("Discover models on Hugging Face", id="hub-summary", classes="muted")
            yield LibraryTable(
                id="hub-table",
                cursor_type="row",
                zebra_stripes=True,
                cursor_foreground_priority="renderable",
            )
            yield Static("Arrows Navigate   Enter Details   Tab Next control", classes="list-hint")
        with VerticalScroll(classes="detail panel", can_focus=True):
            yield Button("Back to results", id="hub-back", classes="back")
            yield Label("Select a model", id="hub-title", classes="detail-title")
            yield Link("huggingface.co", url="https://huggingface.co", id="hub-link")
            yield Label("QUANTIZATION / GGUF", classes="eyebrow")
            yield Select([], prompt="Select a GGUF file", id="hub-quant", disabled=True)
            yield Static("Reading GPU information...", id="hub-gpu", classes="info-card")
            yield Static(
                "Select a file to estimate memory usage.", id="hub-fit", classes="info-card"
            )
            yield ProgressBar(total=100, show_eta=False, show_percentage=False, id="hub-memory")
            yield Button("Download GGUF", variant="primary", id="hub-download", disabled=True)
            with TabbedContent():
                with TabPane("README", id="hub-readme-tab"):
                    yield Markdown(
                        "Select a model to read its model card.", id="hub-readme", open_links=False
                    )
                with TabPane("Details", id="hub-details-tab"):
                    yield Static("", id="hub-metadata")
                with TabPane("Files", id="hub-files-tab"):
                    yield Static("", id="hub-files")

    def refresh_view(self) -> None:
        self.run_action(self.search, group="search")

    @on(Button.Pressed, "#hub-search")
    @on(Input.Submitted, "#hub-query")
    @on(Select.Changed, "#hub-sort")
    def search_requested(self) -> None:
        self.refresh_view()

    async def search(self) -> None:
        summary = self.query_one("#hub-summary", Static)
        summary.update("Searching Hugging Face...")
        query = self.query_one("#hub-query", Input).value.strip()
        sort = str(self.query_one("#hub-sort", Select).value)
        try:
            data = await self.client.get("/api/models/search", q=query, sort=sort)
        except Exception:
            summary.update("Search unavailable. Use Search to retry.")
            raise
        self.loaded = True
        self.workers.cancel_group(self, "details")
        self.repo_id = ""
        self.details = {}
        self.files = {}
        self.query_one("#hub-quant", Select).set_options([])
        self.query_one("#hub-quant", Select).disabled = True
        self.query_one("#hub-title", Label).update("Select a model")
        self.query_one("#hub-metadata", Static).update("")
        self.query_one("#hub-files", Static).update("")
        self.update_memory()
        self.models = data.get("models", [])
        self.render_table()
        summary.update(
            f"{len(self.models)} results / Hugging Face GGUF"
            if self.models
            else "No results returned. Try another search or retry."
        )
        if self.models:
            self.select_repo(self.models[0]["id"])
        else:
            await self.query_one("#hub-readme", Markdown).update("No model selected.")

    @on(LibraryTable.Resized)
    def render_table(self) -> None:
        table = self.query_one("#hub-table", DataTable)
        table.clear(columns=True)
        width = self.app.size.width
        available = table.size.width or (width - 8 if width < 110 else width - 71)
        extra = (18 if width >= 155 else 0) + (30 if width >= 185 else 0)
        table.add_column("Model", width=max(16, min(52, available - 25 - extra)))
        if width >= 155:
            table.add_column("Author", width=16)
        table.add_column("Downloads", width=11)
        table.add_column("Likes", width=7)
        if width >= 185:
            table.add_column("Capabilities", width=28)
        for model in self.models:
            repo = model["id"]
            row: list[Any] = [Text(repo.split("/")[-1], overflow="ellipsis", no_wrap=True)]
            if width >= 155:
                row.append(
                    Text(
                        model.get("author") or repo.split("/")[0], overflow="ellipsis", no_wrap=True
                    )
                )
            row.extend([f"{model.get('downloads', 0):,}", f"{model.get('likes', 0):,}"])
            if width >= 185:
                row.append(capabilities(model, self.app.palette))
            table.add_row(*row, key=repo, height=2 if self.app.size.height >= 35 else 1)
        if self.repo_id in {key.value for key in table.rows}:
            table.move_cursor(row=table.get_row_index(self.repo_id))

    def refresh_palette(self) -> None:
        table = self.query_one("#hub-table", DataTable)
        if len(table.columns) < 5:
            return  # The capabilities column is hidden in smaller terminals.
        for model in self.models:
            if model["id"] in table.rows:
                table.update_cell_at(
                    (table.get_row_index(model["id"]), 4), capabilities(model, self.app.palette)
                )

    @on(DataTable.RowHighlighted, "#hub-table")
    def highlighted(self, event: DataTable.RowHighlighted) -> None:
        table = self.query_one("#hub-table", DataTable)
        if not table.row_count or table.ordered_rows[table.cursor_row].key != event.row_key:
            return
        self.select_repo(str(event.row_key.value))

    @on(DataTable.RowSelected, "#hub-table")
    def selected(self, event: DataTable.RowSelected) -> None:
        self.select_repo(str(event.row_key.value))
        self.show_details()

    def select_repo(self, repo: str) -> None:
        if repo == self.repo_id:
            return
        self.repo_id = repo
        self.details = {}
        self.files = {}
        quant = self.query_one("#hub-quant", Select)
        quant.set_options([])
        quant.disabled = True
        self.query_one("#hub-download", Button).disabled = True
        self.query_one("#hub-title", Label).update(Text(repo.split("/")[-1]))
        link = self.query_one("#hub-link", Link)
        link.text, link.url = repo, f"https://huggingface.co/{repo}"
        self.run_action(self.load_details, repo, group="details")

    async def load_details(self, repo: str) -> None:
        await asyncio.sleep(0.2)  # Debounce arrow-key navigation through remote results.
        readme = self.query_one("#hub-readme", Markdown)
        await readme.update("Loading model card...")
        try:
            data = await self.client.get("/api/models/hf-details", repo_id=repo)
        except Exception:
            await readme.update("Could not load this model card. Search again to retry.")
            self.repo_id = ""
            raise
        if repo != self.repo_id:
            return
        self.details = data.get("details") or {}
        siblings = self.details.get("siblings") or []
        self.files = {file["rfilename"]: file for file in siblings if file.get("rfilename")}
        names = sorted(name for name in self.files if name.lower().endswith(".gguf"))
        quant = self.query_one("#hub-quant", Select)
        quant.set_options([(Text(name), name) for name in names])
        quant.disabled = not names
        if names:
            quant.value = next((name for name in names if "q4_k_m" in name.lower()), names[0])
        self.query_one("#hub-files", Static).update(
            Text(
                "\n".join(
                    f"{name}\n  {size_label(file.get('size') or (file.get('lfs') or {}).get('size'))}"
                    for name, file in self.files.items()
                )
                or "No files listed."
            )
        )
        metadata = self.details
        self.query_one("#hub-metadata", Static).update(
            Text(
                f"Repository: {repo}\n\nDownloads: {metadata.get('downloads', 0):,}\n"
                f"Likes: {metadata.get('likes', 0):,}\n"
                f"Task: {metadata.get('pipeline_tag', 'Not specified')}\n"
                f"License: {(metadata.get('cardData') or {}).get('license', 'Not specified')}\n\n"
                f"Tags: {', '.join(metadata.get('tags') or [])}"
            )
        )
        # Model cards often start with YAML and contain HTML/images. Keep the readable Markdown.
        markdown = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", data.get("readme") or "", flags=re.S)
        await readme.update(markdown or "No README available.")
        self.update_memory()

    @on(Select.Changed, "#hub-quant")
    def quant_changed(self) -> None:
        self.update_memory()

    def update_memory(self) -> None:
        gpu = self.app.gpu
        memory = gpu.get("vram", 0)
        self.query_one("#hub-gpu", Static).update(
            Text(f"{gpu.get('name', 'GPU unavailable')}\nReported memory: {memory} GiB")
        )
        name = str(self.query_one("#hub-quant", Select).value)
        file = self.files.get(name, {})
        size = file.get("size") or (file.get("lfs") or {}).get("size") or 0
        split = bool(re.search(r"-\d{5}-of-\d{5}\.gguf$", name, re.I))
        text = f"Weights: {size_label(size)}"
        if split:
            text += "\nSplit model: download every shard via the Hub."
        elif size and memory:
            text += f" / {size / (memory * 1024**3):.0%} of reported memory"
        text += "\nEstimate only; context cache and runtime need additional memory."
        self.query_one("#hub-fit", Static).update(Text(text))
        self.query_one("#hub-memory", ProgressBar).update(
            progress=min(100, size / (memory * 1024**3) * 100) if memory else 0
        )
        self.query_one("#hub-download", Button).disabled = (
            not file or split or self.downloading or self.app.download_active
        )

    @on(Button.Pressed, "#hub-download")
    def request_download(self) -> None:
        repo, filename = self.repo_id, str(self.query_one("#hub-quant", Select).value)
        if filename not in self.files:
            return
        self.app.push_screen(
            Confirm(
                "Download GGUF",
                f"{repo}\n{filename}\n\nSave in your configured model directory?",
                "Download",
            ),
            lambda confirmed: self.run_action(self.download, repo, filename) if confirmed else None,
        )

    async def download(self, repo: str, filename: str) -> None:
        self.downloading = True
        self.update_memory()
        try:
            await self.client.post("/api/models/download", repo_id=repo, filename=filename)
            await self.app.poll_status()
            self.app.notify("Download started. Progress stays visible across all sections.")
        finally:
            self.downloading = False
            self.update_memory()

    @on(Button.Pressed, "#hub-back")
    def back(self) -> None:
        self.show_details(False)
