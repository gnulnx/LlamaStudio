"""Saved conversations with streaming Markdown, reasoning, and tool activity."""

from __future__ import annotations

import json
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Checkbox, Collapsible, Label, Markdown, OptionList, Static
from textual.widgets.option_list import Option

from .client import APIError, error_message
from .widgets import Composer, Confirm, Rename, StudioView


class MessageCard(Vertical):
    def __init__(self, role: str, content: str = "", **kwargs: Any):
        super().__init__(classes=f"message {role}", **kwargs)
        self.role, self.initial_content = role, content
        self.last_kind = ""
        self.last_markdown: Markdown | None = None
        self.reasoning_box: Collapsible | None = None

    def compose(self) -> ComposeResult:
        yield Label(self.role.upper(), classes="message-role")
        if self.initial_content:
            # User input is literal text; model responses retain Markdown formatting.
            yield (
                Static(Text(self.initial_content))
                if self.role == "user"
                else Markdown(self.initial_content, open_links=False)
            )

    async def append_piece(self, kind: str, content: str) -> None:
        if kind != self.last_kind or self.last_markdown is None:
            markdown = Markdown("", open_links=False)
            if kind == "reasoning":
                self.reasoning_box = Collapsible(
                    markdown, title="Reasoning", collapsed=False, classes="reasoning-activity"
                )
                await self.mount(self.reasoning_box)
            else:
                if self.reasoning_box is not None:
                    self.reasoning_box.collapsed = True
                await self.mount(markdown)
            self.last_kind, self.last_markdown = kind, markdown
        await self.last_markdown.append(content)

    async def add_tool(self, name: str, arguments: Any, result: Any = None) -> None:
        self.last_kind = "tool"
        if self.reasoning_box is not None:
            self.reasoning_box.collapsed = True
        text = (
            json.dumps(arguments, indent=2, ensure_ascii=False)
            if not isinstance(arguments, str)
            else arguments
        )
        if result is not None:
            text += f"\n\nResult\n{result}"
        await self.mount(
            Collapsible(
                Static(Text(text)), title=f"Tool / {name}", collapsed=True, classes="tool-activity"
            )
        )


class ChatView(StudioView):
    def __init__(self):
        super().__init__(id="chat")
        self.conversations: list[dict[str, Any]] = []
        self.conversation_id = ""
        self.loaded = False
        self.busy = False
        self.streaming = False

    def compose(self) -> ComposeResult:
        with Vertical(classes="master panel conversations"):
            yield Label("CONVERSATIONS", classes="eyebrow")
            yield Button("+ New chat", id="chat-new", variant="primary")
            yield OptionList(id="chat-conversations")
            with Horizontal(classes="actions"):
                yield Button("Rename", id="chat-rename", disabled=True)
                yield Button("Delete", id="chat-delete", disabled=True)
        with Vertical(classes="detail panel chat-detail"):
            with Horizontal(classes="chat-heading"):
                yield Button("Chats", id="chat-back", classes="back")
                yield Label("Start a conversation", id="chat-title", classes="detail-title")
            yield VerticalScroll(
                Static(
                    "Your local workspace for ideas.\n\nLoad a model in Models, then write a message below.",
                    id="chat-welcome",
                    classes="welcome",
                ),
                id="chat-transcript",
            )
            yield Static("Saved on this LlamaStudio backend", id="chat-state", classes="muted")
            yield Composer(
                id="chat-input", placeholder="Message your model...", show_line_numbers=False
            )
            with Horizontal(classes="chat-controls"):
                yield Checkbox("Thinking", True, id="chat-thinking")
                yield Static("Enter New line / Ctrl+S Send", classes="composer-hint")
                yield Button("Send", id="chat-send", variant="primary")

    def refresh_view(self) -> None:
        if not self.busy:
            self.run_action(self.fetch_conversations, group="conversation")

    async def fetch_conversations(self) -> None:
        await self.refresh_conversations()
        if self.conversations:
            conversation = next(
                (conv for conv in self.conversations if conv["id"] == self.conversation_id), None
            )
            if conversation is None:
                conversation = next(
                    (conv for conv in self.conversations if conv.get("is_active")),
                    self.conversations[0],
                )
            await self.open_conversation(conversation["id"])
        else:
            self.conversation_id = ""
            await self.query_one("#chat-transcript", VerticalScroll).remove_children()
            self.query_one("#chat-title", Label).update("Start a conversation")
            self.update_controls()
        self.loaded = True

    async def refresh_conversations(self) -> None:
        data = await self.client.get("/api/chat/conversations")
        self.conversations = list(reversed(data.get("conversations", [])))
        options = self.query_one("#chat-conversations", OptionList)
        options.clear_options()
        options.add_options([
            Option(Text(f"{conv['title']}\n{conv.get('message_count', 0)} messages"), id=conv["id"])
            for conv in self.conversations
        ])
        for index, conv in enumerate(self.conversations):
            if conv["id"] == self.conversation_id:
                options.highlighted = index
                self.query_one("#chat-title", Label).update(Text(conv["title"]))
                break
        self.update_controls()

    def update_controls(self) -> None:
        for identifier in (
            "chat-new",
            "chat-conversations",
            "chat-input",
            "chat-send",
            "chat-thinking",
        ):
            self.query_one(f"#{identifier}").disabled = self.busy
        for identifier in ("chat-rename", "chat-delete"):
            self.query_one(f"#{identifier}").disabled = self.busy or not self.conversation_id

    @on(OptionList.OptionSelected, "#chat-conversations")
    def conversation_selected(self, event: OptionList.OptionSelected) -> None:
        if not self.busy and event.option.id:
            self.run_action(self.open_conversation, event.option.id, group="conversation")
            self.show_details()

    async def open_conversation(self, conversation_id: str) -> None:
        self.busy = True
        self.update_controls()
        try:
            data = await self.client.post(f"/api/chat/switch/{conversation_id}")
            self.conversation_id = conversation_id
            transcript = self.query_one("#chat-transcript", VerticalScroll)
            await transcript.remove_children()
            for message in data.get("messages", []):
                role = message.get("role", "assistant")
                card = MessageCard(role)
                await transcript.mount(card)
                if message.get("reasoning"):
                    await card.mount(
                        Collapsible(
                            Markdown(message["reasoning"], open_links=False),
                            title="Reasoning",
                            collapsed=True,
                        )
                    )
                content = message.get("content") or ""
                if content:
                    await card.mount(
                        Static(Text(content))
                        if role in {"user", "tool"}
                        else Markdown(content, open_links=False)
                    )
                for tool in message.get("tool_calls") or []:
                    function = tool.get("function") or {}
                    await card.add_tool(function.get("name", "Tool"), function.get("arguments", {}))
                if message.get("images") or message.get("audios"):
                    await card.mount(
                        Static("Media attachment / view in the web app", classes="muted")
                    )
                if message.get("error"):
                    await card.mount(
                        Static(Text(error_message(message["error"])), classes="error-text")
                    )
            conv = next((conv for conv in self.conversations if conv["id"] == conversation_id), {})
            self.query_one("#chat-title", Label).update(Text(conv.get("title", "New chat")))
            self.query_one("#chat-state", Static).update("Saved conversation / ready")
            transcript.anchor()
        finally:
            self.busy = False
            self.update_controls()

    @on(Button.Pressed, "#chat-new")
    def new_requested(self) -> None:
        if not self.busy:
            self.run_action(self.new_conversation, group="conversation")

    async def new_conversation(self) -> None:
        self.busy = True
        self.update_controls()
        try:
            data = await self.client.post("/api/chat/new")
            self.conversation_id = data["id"]
            await self.refresh_conversations()
            await self.query_one("#chat-transcript", VerticalScroll).remove_children()
            self.query_one("#chat-title", Label).update("New chat")
            self.query_one("#chat-input", Composer).clear()
            self.query_one("#chat-state", Static).update("New conversation / ready")
            self.show_details()
        finally:
            self.busy = False
            self.update_controls()
            self.query_one("#chat-input", Composer).focus()

    @on(Button.Pressed, "#chat-send")
    def send_message(self) -> None:
        if self.busy:
            return
        message = self.query_one("#chat-input", Composer).text.strip()
        if not message:
            return
        self.busy = True  # Lock immediately, including rapid double-clicks / repeated keys.
        self.update_controls()
        self.run_action(self.stream_message, message, group="chat-stream", exclusive=False)

    async def stream_message(self, message: str) -> None:
        self.streaming = True
        transcript = self.query_one("#chat-transcript", VerticalScroll)
        status = self.query_one("#chat-state", Static)
        card: MessageCard | None = None
        try:
            if not self.conversation_id:
                data = await self.client.post("/api/chat/new")
                self.conversation_id = data["id"]
                await transcript.remove_children()
            else:
                await self.client.post(f"/api/chat/switch/{self.conversation_id}")
            welcome = transcript.query("#chat-welcome")
            if welcome:
                await welcome.remove()
            await transcript.mount(MessageCard("user", message))
            card = MessageCard("assistant")
            await transcript.mount(card)
            self.query_one("#chat-input", Composer).clear()
            status.update("Generating...")
            transcript.anchor()
            had_metrics = False
            async for event in self.client.chat(
                message=message, enable_thinking=self.query_one("#chat-thinking", Checkbox).value
            ):
                if event.get("error"):
                    raise APIError(error_message(event["error"]))
                for kind in ("reasoning", "content"):
                    if event.get(kind):
                        await card.append_piece(kind, event[kind])
                if event.get("type") == "tool_exec_start":
                    status.update(Text(f"Running tool / {event.get('name', 'tool')}"))
                    await card.add_tool(event.get("name", "tool"), event.get("arguments", {}))
                elif event.get("type") == "tool_exec_end":
                    await card.mount(
                        Collapsible(
                            Static(Text(str(event.get("result", "")))),
                            title=f"Result / {event.get('name', 'tool')}",
                            collapsed=True,
                        )
                    )
                    status.update("Generating...")
                elif event.get("type") == "metrics":
                    had_metrics = True
                    metrics = event.get("metrics") or {}
                    status.update(
                        f"{metrics.get('completion_tokens', 0)} tokens / {metrics.get('tokens_per_second', 0):.1f} tokens/s"
                    )
            if not had_metrics:
                status.update("Response complete / saved")
        except APIError as exc:
            status.update("Response interrupted / see error below")
            if card is not None:
                await card.mount(Static(Text(str(exc)), classes="error-text"))
            else:
                self.app.notify(str(exc), severity="error")
            # Keep a copy available for editing/retry even when the server rejected the turn.
            self.query_one("#chat-input", Composer).load_text(message)
        finally:
            self.busy = self.streaming = False
            self.update_controls()
            try:
                await self.refresh_conversations()
            except APIError:
                self.app.notify(
                    "Could not refresh saved conversations. Use Ctrl+R to retry.",
                    severity="warning",
                )
            if self.app.current_view == "chat":
                self.query_one("#chat-input", Composer).focus()

    @on(Button.Pressed, "#chat-rename")
    def rename_requested(self) -> None:
        conversation_id = self.conversation_id
        conv = next((conv for conv in self.conversations if conv["id"] == conversation_id), {})
        self.app.push_screen(
            Rename(conv.get("title", "")),
            lambda title: self.run_action(self.rename, conversation_id, title) if title else None,
        )

    async def rename(self, conversation_id: str, title: str) -> None:
        await self.client.post(f"/api/chat/rename/{conversation_id}", title=title)
        await self.refresh_conversations()

    @on(Button.Pressed, "#chat-delete")
    def delete_requested(self) -> None:
        conversation_id = self.conversation_id
        self.app.push_screen(
            Confirm(
                "Delete conversation",
                "Permanently delete this saved conversation and its messages?",
                "Delete",
            ),
            lambda yes: self.run_action(self.delete, conversation_id) if yes else None,
        )

    async def delete(self, conversation_id: str) -> None:
        await self.client.request("DELETE", f"/api/chat/{conversation_id}")
        self.conversation_id = ""
        await self.query_one("#chat-transcript", VerticalScroll).remove_children()
        self.query_one("#chat-title", Label).update("Start a conversation")
        await self.fetch_conversations()

    @on(Button.Pressed, "#chat-back")
    def back(self) -> None:
        self.show_details(False)
