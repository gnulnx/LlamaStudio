"""Shared terminal controls and presentation helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Label, Markdown, Static, TextArea

from .client import APIError, StudioClient


def size_label(value: int | float | None) -> str:
    if not value:
        return "Size unknown"
    return f"{value / 1024**3:.2f} GiB"


class LibraryTable(DataTable):
    """Notify the owning view after the table's actual available width changes."""

    class Resized(Message):
        pass

    def on_resize(self) -> None:
        self.post_message(self.Resized())


class StudioView(Horizontal):
    @property
    def client(self) -> StudioClient:
        return self.app.client

    def run_action(
        self,
        function: Callable[..., Awaitable[Any]],
        *args: Any,
        group: str = "action",
        exclusive: bool = True,
    ) -> None:
        async def guarded() -> None:
            try:
                await function(*args)
            except APIError as exc:
                self.app.notify(str(exc), title="LlamaStudio", severity="error", timeout=8)

        self.run_worker(guarded, group=group, exclusive=exclusive)

    def show_details(self, show: bool = True) -> None:
        self.set_class(show, "show-detail")
        target = self.query_one(".detail" if show else ".master")
        if not show:
            for control in target.query("DataTable, OptionList"):
                target = control
                break
        elif self.id == "chat":
            target = self.query_one("#chat-input")
        self.app.call_after_refresh(target.focus)


class Confirm(ModalScreen[bool]):
    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def __init__(self, title: str, message: str, confirm: str = "Confirm"):
        super().__init__()
        self.title_text, self.message, self.confirm_text = title, message, confirm

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(Text(self.title_text), classes="dialog-title")
            yield Static(Text(self.message))
            with Horizontal(classes="actions"):
                yield Button("Cancel", id="cancel")
                yield Button(self.confirm_text, id="confirm", variant="primary")

    @on(Button.Pressed)
    def choose(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")

    def action_cancel(self) -> None:
        self.dismiss(False)


class Rename(ModalScreen[str | None]):
    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def __init__(self, title: str):
        super().__init__()
        self.old_title = title

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label("Rename conversation", classes="dialog-title")
            yield Input(self.old_title, id="title-input", max_length=200)
            with Horizontal(classes="actions"):
                yield Button("Cancel", id="cancel")
                yield Button("Save", id="save", variant="primary")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    @on(Input.Submitted)
    @on(Button.Pressed, "#save")
    def save(self) -> None:
        title = self.query_one(Input).value.strip()
        if title:
            self.dismiss(title)

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)


class Help(ModalScreen[None]):
    BINDINGS: ClassVar = [("escape", "dismiss", "Close"), ("f1", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog help-dialog"):
            yield Label("LlamaStudio / Keyboard guide", classes="dialog-title")
            with VerticalScroll():
                yield Markdown(
                    "| Key | Action |\n| --- | --- |\n"
                    "| F2 / F3 / F4 / F5 | Discover / Models / Chat / Logs |\n"
                    "| Tab / Shift+Tab | Next / previous control |\n"
                    "| Arrows, PageUp, PageDown | Navigate lists and text |\n"
                    "| Enter / Space | Activate the focused control |\n"
                    "| Escape | Back from details on a small screen |\n"
                    "| Ctrl+R | Refresh the current section |\n"
                    "| Ctrl+N | New conversation (in Chat) |\n"
                    "| Ctrl+S | Send a chat message |\n"
                    "| Enter (chat editor) | New line |\n"
                    "| Ctrl+Q | Close the TUI |\n\n"
                    "Mouse clicks and scrolling work too. Select text to copy using your "
                    "terminal's copy shortcut; hold Shift if your terminal requires it.\n\n"
                    "Quitting leaves the backend, loaded model, and downloads running. "
                    "The web app and TUI share saved conversations and model settings. "
                    "Images, audio, and microphone input are available in the web app."
                )
            yield Button("Close", id="close", variant="primary")

    @on(Button.Pressed, "#close")
    def close_help(self) -> None:
        self.dismiss()


class Composer(TextArea):
    """Multiline input with a send shortcut supported by ordinary terminals."""

    BINDINGS: ClassVar = [("ctrl+s", "send", "Send")]

    def action_send(self) -> None:
        self.app.query_one("#chat").send_message()
