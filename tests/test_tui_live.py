"""Opt-in TUI smoke test using the running backend and its loaded model.

RUN_TUI_LIVE=1 python -m pytest tests/test_tui_live.py -q -s
Creates its own conversation, renders review SVGs in .runtime/tui, then removes
only that test conversation and restores the previously selected conversation.
"""

from __future__ import annotations

import asyncio
import os
import unittest

from textual.widgets import Checkbox, Markdown

from app.cli import API_BASE_URL
from app.tools import check_path_safe
from app.tui.application import StudioApp
from app.tui.chat import ChatView
from app.tui.client import StudioClient
from app.tui.widgets import Composer


@unittest.skipUnless(
    os.environ.get("RUN_TUI_LIVE") == "1", "Requires a running backend and loaded model"
)
class TestLiveTUI(unittest.IsolatedAsyncioTestCase):
    async def test_real_model_chat_and_persistence(self):
        client = StudioClient(API_BASE_URL)
        conversation_id = None
        previous_id = None
        try:
            status = await client.get("/api/server/status")
            self.assertTrue(status.get("running"), "Load a model with lls load before this test.")
            previous = await client.get("/api/chat/conversations")
            previous_id = next(
                (conv["id"] for conv in previous["conversations"] if conv.get("is_active")), None
            )
            conversation = await client.post("/api/chat/new")
            conversation_id = conversation["id"]
            await client.post(
                f"/api/chat/rename/{conversation_id}", title="TUI integration smoke test"
            )
            app = StudioApp(API_BASE_URL, initial_view="chat")
            async with app.run_test(size=(180, 48)) as pilot:
                await asyncio.wait_for(app.workers.wait_for_complete(), timeout=30)
                await pilot.pause()
                chat = app.query_one(ChatView)
                self.assertEqual(chat.conversation_id, conversation_id)
                app.query_one("#chat-thinking", Checkbox).value = True
                editor = app.query_one("#chat-input", Composer)
                editor.load_text(
                    "Use list_dir to list the current workspace directory (.) once. Then say: LlamaStudio TUI is connected. Keep the reply brief. Do not read or change any files."
                )
                editor.focus()
                await pilot.press("ctrl+s")
                await asyncio.wait_for(app.workers.wait_for_complete(), timeout=120)
                await pilot.pause()
                content = "\n".join(widget.source for widget in chat.query(Markdown))
                self.assertIn("LlamaStudio TUI is connected", content)
                self.assertFalse(chat.streaming)
                self.assertFalse(chat.query(".error-text"))
                screenshot = check_path_safe(".runtime/tui/chat-live.svg")
                screenshot.parent.mkdir(parents=True, exist_ok=True)
                app.save_screenshot(filename=str(screenshot))
                await pilot.resize_terminal(80, 24)
                chat.show_details()
                await pilot.pause()
                app.save_screenshot(
                    filename=str(check_path_safe(".runtime/tui/chat-small-live.svg"))
                )
            saved = await client.post(f"/api/chat/switch/{conversation_id}")
            self.assertTrue(
                any(
                    message["role"] == "assistant" and message.get("content")
                    for message in saved["messages"]
                )
            )
            self.assertTrue(
                any(message.get("tool_calls") for message in saved["messages"]),
                "Expected real tool activity",
            )
            self.assertTrue(
                any(message.get("reasoning") for message in saved["messages"]),
                "Expected streamed reasoning",
            )
        finally:
            if conversation_id:
                await client.request("DELETE", f"/api/chat/{conversation_id}")
            if previous_id:
                await client.post(f"/api/chat/switch/{previous_id}")
            await client.close()
