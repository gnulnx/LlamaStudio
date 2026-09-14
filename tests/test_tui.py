"""Exercise actual Textual widgets against an isolated HTTP contract fixture."""

from __future__ import annotations

import asyncio
import json
import threading
import unittest
from dataclasses import replace
from unittest.mock import patch

import httpx
from textual.widgets import Button, DataTable, Input, Markdown, Select, Static

from app.tui.application import StudioApp
from app.tui.chat import ChatView, MessageCard
from app.tui.client import StudioClient
from app.tui.discover import DiscoverView
from app.tui.themes import BundledThemeAdapter, ResolvedTheme, ThemeAdapter
from app.tui.widgets import Composer, Confirm


class Backend:
    """Stateful fixture exercising the same routes used by the web app."""

    def __init__(self):
        self.calls = []
        self.online = True
        self.fail_chat = False
        self.models = [
            {
                "path": "/models/tiny.gguf",
                "name": "Tiny Q4_K_M",
                "quant": "Q4_K_M",
                "size_human": "1 GiB",
                "size": 1024**3,
            }
        ]
        self.profiles = {
            "/models/tiny.gguf": {
                "ctx_size": 8192,
                "gpu_layers": 42,
                "threads": 6,
                "custom_template": "keep me",
                "inference_settings": {"temperature": 0.7},
            }
        }
        self.status = {
            "running": True,
            "current_model": "/models/tiny.gguf",
            "current_model_name": "Tiny Q4_K_M",
            "current_params": {"gpu_layers": 42, "cpu_mode": False},
        }
        self.gpu = {"name": "Test GPU", "vram": 24, "total_vram": 24, "used_vram": 6}
        self.gpu_available = True
        self.progress = {"status": "idle"}
        self.conversations = {
            "saved": {
                "id": "saved",
                "title": "Saved [conversation]",
                "is_active": True,
                "messages": [
                    {"role": "user", "content": "Remember this"},
                    {
                        "role": "assistant",
                        "content": "I remember.",
                        "reasoning": "Earlier thinking",
                    },
                ],
            }
        }
        self.active_id = "saved"
        self.chat_gate: asyncio.Event | None = None
        self.download_gate: asyncio.Event | None = None
        self.download_requested = asyncio.Event()

    async def __call__(self, request):
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        self.calls.append((request.method, path, body, dict(request.url.params)))
        if not self.online:
            raise httpx.ConnectError("offline", request=request)
        data = {}
        if path == "/api/server/status":
            data = self.status
        elif path == "/api/gpu":
            if not self.gpu_available:
                return httpx.Response(503, json={"detail": "Telemetry unavailable"})
            data = self.gpu
        elif path == "/api/models/search":
            data = {
                "models": [
                    {
                        "id": "org/Tiny-GGUF",
                        "author": "org",
                        "likes": 42,
                        "downloads": 10000,
                        "tags": ["tools", "reasoning"],
                    },
                    {"id": "org/Second-GGUF", "likes": 12, "downloads": 42},
                ]
            }
        elif path == "/api/models/hf-details":
            repo = request.url.params["repo_id"]
            data = {
                "details": {
                    "id": repo,
                    "downloads": 42,
                    "likes": 12,
                    "siblings": [
                        {"rfilename": "tiny-Q4_K_M.gguf", "size": 1024**3},
                        {"rfilename": "tiny-00001-of-00002.gguf", "size": 1024**3},
                        {"rfilename": "README.md", "size": 100},
                    ],
                },
                "readme": f"---\nlicense: apache-2.0\n---\n# {repo}\n\nA **real widget** model card.",
            }
        elif path in ("/api/models", "/api/models/refresh"):
            data = {"models": self.models}
        elif path == "/api/models/settings":
            if request.method == "POST":
                self.profiles[body["path"]] = body["settings"]
            else:
                data = self.profiles
        elif path == "/api/models/load":
            self.status.update(
                running=True, current_model=body["path"], current_model_name="Tiny Q4_K_M"
            )
            self.profiles[body["path"]] = body["settings"]
        elif path == "/api/models/eject":
            self.status.update(running=False, current_model=None, current_model_name=None)
        elif path == "/api/models/delete":
            self.models = [model for model in self.models if model["path"] != body["path"]]
        elif path == "/api/models/download":
            self.progress = {
                "status": "downloading",
                "filename": body["filename"],
                "percent": 25,
                "total_bytes": 1024,
                "speed_mb": 12,
            }
        elif path == "/api/models/download/active":
            if self.download_gate:
                self.download_requested.set()
                await self.download_gate.wait()
            data = {"active": self.progress["status"] == "downloading", "progress": self.progress}
        elif path == "/api/models/download/cancel":
            self.progress["status"] = "cancelled"
        elif path == "/api/server/logs":
            data = {"logs": ["INFO Server ready\n", "ERROR Example failure\n"]}
        elif path == "/api/chat/conversations":
            data = {
                "conversations": [
                    {key: value for key, value in conv.items() if key != "messages"}
                    | {"message_count": len(conv["messages"])}
                    for conv in self.conversations.values()
                ]
            }
        elif path.startswith("/api/chat/switch/"):
            self.active_id = path.split("/")[-1]
            data = self.conversations[self.active_id]
        elif path == "/api/chat/new":
            self.active_id = f"new-{len(self.conversations)}"
            data = {"id": self.active_id, "title": "New Chat", "messages": []}
            self.conversations[self.active_id] = data
        elif path.startswith("/api/chat/rename/"):
            self.conversations[path.split("/")[-1]]["title"] = body["title"]
        elif path == "/api/chat/send":
            if self.chat_gate:
                await self.chat_gate.wait()
            if self.fail_chat:
                return httpx.Response(503, json={"detail": "Load a model first"})
            self.conversations[self.active_id]["messages"].extend([
                {"role": "user", "content": body["message"]},
                {"role": "assistant", "content": "Hello **world**."},
            ])
            events = [
                {"type": "start"},
                {"reasoning": "Let me think."},
                {
                    "type": "tool_exec_start",
                    "name": "read_file",
                    "arguments": {"file_path": "README.md"},
                },
                {"type": "tool_exec_end", "name": "read_file", "result": "File contents"},
                {"content": "Hello "},
                {"content": "**world**."},
                {"type": "metrics", "metrics": {"completion_tokens": 3, "tokens_per_second": 42.0}},
                {"type": "end"},
            ]
            return httpx.Response(
                200, text="".join(f"data: {json.dumps(event)}\n\n" for event in events)
            )
        elif request.method == "DELETE" and path.startswith("/api/chat/"):
            del self.conversations[path.split("/")[-1]]
        else:
            raise AssertionError(f"Unexpected API request: {request.method} {path}")
        return httpx.Response(200, json=data)


class TestTUI(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        system = patch("app.tui.themes.SYSTEM_ADAPTERS", ())
        system.start()
        self.addCleanup(system.stop)
        self.backend = Backend()
        self.app = StudioApp(
            "http://studio",
            client=StudioClient("http://studio", transport=httpx.MockTransport(self.backend)),
        )

    async def settle(self, pilot):
        for _ in range(3):
            await pilot.pause(0.12)
            await self.app.workers.wait_for_complete()

    async def test_discovery_mouse_and_keyboard_with_real_details(self):
        async with self.app.run_test(size=(190, 52)) as pilot:
            await self.settle(pilot)
            self.assertEqual(self.app.query_one("#hub-table", DataTable).row_count, 2)
            self.assertIn("Tiny-GGUF", self.app.query_one("#hub-readme", Markdown).source)
            self.assertFalse(self.app.query_one("#hub-download", Button).disabled)
            await pilot.click("#hub-query")
            await pilot.press("q", "w", "e", "n", "enter")
            await self.settle(pilot)
            searches = [call for call in self.backend.calls if call[1] == "/api/models/search"]
            self.assertEqual(searches[-1][3]["q"], "qwen")
            self.app.query_one("#hub-table").focus()
            await pilot.press("down", "enter")
            await self.settle(pilot)
            self.assertEqual(self.app.query_one(DiscoverView).repo_id, "org/Second-GGUF")
            await pilot.click("#nav-logs")
            await self.settle(pilot)
            self.assertEqual(self.app.current_view, "logs")
            self.assertEqual(self.app.query_one("#logs").last_lines[0], "INFO Server ready\n")

    async def test_dropdown_options_are_visible_and_selectable(self):
        async with self.app.run_test(size=(190, 52)) as pilot:
            await self.settle(pilot)
            await pilot.click("#hub-sort")
            await pilot.pause()
            overlay = self.app.query_one("#hub-sort SelectOverlay")
            visible_text = "\n".join(
                overlay.render_line(y).text for y in range(overlay.scrollable_content_region.height)
            )
            for label in ("Most likes", "Downloads", "Updated"):
                self.assertIn(label, visible_text)
            normal = overlay.get_component_rich_style("option-list--option")
            highlighted = overlay.get_component_rich_style("option-list--option-highlighted")
            self.assertNotEqual(normal.color, normal.bgcolor)
            self.assertNotEqual(highlighted.color, highlighted.bgcolor)
            await pilot.press("down", "enter")
            await self.settle(pilot)
            self.assertEqual(self.app.query_one("#hub-sort", Select).value, "downloads")

    async def test_palette_reload_updates_css_and_rich_colors_without_resetting_state(self):
        async with self.app.run_test(size=(190, 52)) as pilot:
            await self.settle(pilot)
            self.app.query_one("#hub-quant", Select).value = "tiny-00001-of-00002.gguf"
            await pilot.press("f4")
            await self.settle(pilot)
            editor = self.app.query_one("#chat-input", Composer)
            editor.load_text("Keep my unfinished draft")
            conversation = self.app.query_one(ChatView).conversation_id
            old = self.app.palette
            updated = replace(
                old, background="#080810", surface="#19192f", success="#33ee88", warning="#ffcc33"
            )
            calls_before = list(self.backend.calls)
            with patch.object(
                self.app.theme_adapter,
                "resolve",
                return_value=ResolvedTheme(updated, True, "default"),
            ):
                await pilot.press("ctrl+p")
                await self.app.workers.wait_for_complete()
                await pilot.pause()
            self.assertIs(self.app.palette, updated)
            self.assertEqual(self.app.screen.styles.background.hex.lower(), updated.background)
            self.assertEqual(editor.styles.background.hex.lower(), updated.surface)
            self.assertEqual(editor.text, "Keep my unfinished draft")
            self.assertEqual(self.app.current_view, "chat")
            self.assertEqual(self.app.query_one(ChatView).conversation_id, conversation)
            self.assertEqual(
                self.app.query_one("#hub-quant", Select).value, "tiny-00001-of-00002.gguf"
            )
            badge = self.app.query_one("#hub-table", DataTable).get_cell_at((0, 4))
            self.assertTrue(any(updated.warning in span.style for span in badge.renderable.spans))
            state_colors = [
                segment.style.color.name
                for segment in self.app.query_one("#connection").render_line(0)
                if segment.style and segment.style.color
            ]
            self.assertIn(updated.success, state_colors)
            # Only normal status polling may have made HTTP requests during repaint.
            for method, path, _, _ in self.backend.calls[len(calls_before) :]:
                self.assertEqual(method, "GET")
                self.assertIn(
                    path, ("/api/server/status", "/api/gpu", "/api/models/download/active")
                )

    async def test_invalid_palette_reload_keeps_last_good_colors_and_open_dialog(self):
        async with self.app.run_test(size=(80, 24)) as pilot:
            await self.settle(pilot)
            await pilot.press("f1")
            dialog = self.app.screen
            original = self.app.palette
            with (
                patch.object(
                    self.app.theme_adapter, "resolve", side_effect=ValueError("bad color")
                ),
                patch.object(self.app, "notify") as notify,
            ):
                await pilot.press("ctrl+p")
                await self.app.workers.wait_for_complete()
                await pilot.pause()
            self.assertIs(self.app.palette, original)
            self.assertIs(self.app.screen, dialog)
            self.assertIn("Theme unchanged: bad color", notify.call_args.args[0])

    async def test_theme_picker_switches_mode_preserves_draft_and_cancels(self):
        async with self.app.run_test(size=(80, 24)) as pilot:
            await self.settle(pilot)
            await pilot.press("f4")
            await self.settle(pilot)
            editor = self.app.query_one("#chat-input", Composer)
            editor.load_text("Keep this draft through light and dark")
            for name in ("light", "dark", "slate", "system", "default"):
                await pilot.press("f6")
                self.app.screen.query_one("#theme-choice", Select).value = name
                await pilot.click("#theme-apply")
                await self.app.workers.wait_for_complete()
                await pilot.pause()
                self.assertEqual(self.app.theme_name, name)
                self.assertEqual(self.app.current_theme.dark, name != "light")
                self.assertEqual(self.app.has_class("-light-mode"), name == "light")
                self.assertEqual(editor.text, "Keep this draft through light and dark")
                self.assertEqual(self.app.current_view, "chat")
                self.assertIsNone(self.app._theme_timer)
            await pilot.press("f6")
            self.app.screen.query_one("#theme-choice", Select).value = "light"
            await pilot.press("escape")
            self.assertEqual(self.app.theme_name, "default")

    async def test_system_adapter_auto_refresh_failure_recovery_and_stop(self):
        class Source(ThemeAdapter):
            refresh_interval = 0.1

            def __init__(self):
                self.name = "dark"
                self.fail = False
                self.thread = None

            def resolve(self):
                self.thread = threading.get_ident()
                if self.fail:
                    raise ValueError("theme file being replaced")
                return BundledThemeAdapter(self.name).resolve()

        source = Source()
        with patch("app.tui.themes.SYSTEM_ADAPTERS", (lambda: source,)):
            async with self.app.run_test(size=(80, 24)) as pilot:
                await self.settle(pilot)
                self.app.select_theme("system")
                await self.app.workers.wait_for_complete()
                self.assertIsNotNone(self.app._theme_timer)
                source.name = "light"
                await pilot.pause(0.3)
                self.assertFalse(self.app.current_theme.dark)
                self.assertNotEqual(source.thread, threading.get_ident())
                good = self.app.palette
                source.fail = True
                with patch.object(self.app, "notify") as notify:
                    await pilot.pause(0.4)
                    self.assertIs(self.app.palette, good)
                    notify.assert_called_once()
                source.fail = False
                source.name = "dark"
                await pilot.pause(0.3)
                self.assertTrue(self.app.current_theme.dark)
                self.app.select_theme("default")
                await self.app.workers.wait_for_complete()
                self.assertIsNone(self.app._theme_timer)

    async def test_slow_theme_cannot_overwrite_newer_selection_or_block_input(self):
        started, release = threading.Event(), threading.Event()

        class SlowSource(ThemeAdapter):
            def resolve(self):
                started.set()
                if not release.wait(5):
                    raise ValueError("test source timed out")
                return BundledThemeAdapter("light").resolve()

        async with self.app.run_test(size=(80, 24)) as pilot:
            await self.settle(pilot)
            self.app.request_theme(SlowSource(), "system")
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                await pilot.press("f4")
                self.assertEqual(self.app.current_view, "chat")
                self.app.select_theme("dark")
            finally:
                release.set()
            await self.app.workers.wait_for_complete()
            self.assertEqual(self.app.theme_name, "dark")
            self.assertTrue(self.app.current_theme.dark)

    async def test_download_requires_confirmation_and_survives_navigation(self):
        async with self.app.run_test(size=(180, 48)) as pilot:
            await self.settle(pilot)
            await pilot.click("#hub-download")
            await pilot.pause()
            self.assertIsInstance(self.app.screen, Confirm)
            self.assertFalse(any(call[1] == "/api/models/download" for call in self.backend.calls))
            await pilot.click("#confirm")
            await self.settle(pilot)
            self.assertTrue(self.app.download_active)
            await pilot.press("f3")
            await self.settle(pilot)
            self.assertTrue(self.app.query_one("#download-tray").display)
            await pilot.click("#download-cancel")
            await pilot.pause()
            await pilot.click("#confirm")
            await self.settle(pilot)
            self.assertFalse(self.app.download_active)

    async def test_split_file_cannot_be_mistaken_for_a_complete_model(self):
        async with self.app.run_test(size=(180, 48)) as pilot:
            await self.settle(pilot)
            self.app.query_one("#hub-quant", Select).value = "tiny-00001-of-00002.gguf"
            await pilot.pause()
            self.assertTrue(self.app.query_one("#hub-download", Button).disabled)
            self.assertIn("every shard", str(self.app.query_one("#hub-fit", Static).render()))

    async def test_small_terminal_details_back_and_resize(self):
        async with self.app.run_test(size=(80, 24)) as pilot:
            await self.settle(pilot)
            discover = self.app.query_one(DiscoverView)
            self.assertFalse(discover.query_one(".detail").display)
            self.app.query_one("#hub-table").focus()
            await pilot.press("enter")
            await pilot.pause()
            self.assertTrue(discover.query_one(".detail").display)
            self.assertFalse(discover.query_one(".master").display)
            await pilot.click("#hub-back")
            await pilot.pause()
            self.assertTrue(discover.query_one(".master").display)
            await pilot.resize_terminal(190, 52)
            await pilot.pause()
            self.assertTrue(discover.query_one(".detail").display)
            self.assertTrue(discover.query_one(".master").display)
            await pilot.press("f1")
            await pilot.pause()
            await pilot.press("escape")
            self.assertEqual(len(self.app.screen_stack), 1)

    async def test_load_profile_preserves_unexposed_parameters(self):
        async with self.app.run_test(size=(180, 48)) as pilot:
            await self.settle(pilot)
            await pilot.press("f3")
            await self.settle(pilot)
            self.assertEqual(self.app.query_one("#model-context", Input).value, "8192")
            self.app.query_one("#model-context", Input).value = "32768"
            await pilot.click("#local-load")
            await pilot.pause()
            await pilot.click("#confirm")
            await self.settle(pilot)
            profile = self.backend.profiles["/models/tiny.gguf"]
            self.assertEqual(profile["ctx_size"], 32768)
            self.assertEqual(profile["custom_template"], "keep me")
            self.assertEqual(profile["inference_settings"], {"temperature": 0.7})
            await pilot.click("#local-eject")
            await pilot.pause()
            await pilot.click("#confirm")
            await self.settle(pilot)
            self.assertFalse(self.backend.status["running"])
            self.assertFalse(self.app.query_one("#local-delete", Button).disabled)

    async def test_saved_chat_stream_reasoning_tools_and_new_conversation(self):
        async with self.app.run_test(size=(180, 48)) as pilot:
            await self.settle(pilot)
            await pilot.press("f4")
            await self.settle(pilot)
            chat = self.app.query_one(ChatView)
            self.assertEqual(chat.conversation_id, "saved")
            self.assertEqual(len(chat.query(MessageCard)), 2)
            self.app.query_one("#chat-input", Composer).load_text("Hello")
            self.app.query_one("#chat-input").focus()
            await pilot.press("ctrl+s")
            await self.settle(pilot)
            self.assertFalse(chat.busy)
            self.assertEqual(len(chat.query(MessageCard)), 4)
            markdown = [widget.source for widget in chat.query(Markdown)]
            self.assertIn("Let me think.", markdown)
            self.assertIn("Hello **world**.", markdown)
            self.assertIn("42.0 tokens/s", str(self.app.query_one("#chat-state", Static).render()))
            self.assertEqual(len(chat.query("Collapsible")), 4)
            await pilot.press("ctrl+n")
            await self.settle(pilot)
            self.assertTrue(chat.conversation_id.startswith("new-"))
            self.assertEqual(len(chat.query(MessageCard)), 0)

    async def test_chat_error_restores_editor(self):
        async with self.app.run_test(size=(80, 24)) as pilot:
            await self.settle(pilot)
            await pilot.press("f4")
            await self.settle(pilot)
            await pilot.click("#chat-new")
            await self.settle(pilot)
            self.backend.fail_chat = True
            editor = self.app.query_one("#chat-input", Composer)
            editor.load_text("Keep this draft")
            editor.focus()
            await pilot.press("ctrl+s")
            await self.settle(pilot)
            self.assertEqual(editor.text, "Keep this draft")
            self.assertFalse(editor.disabled)
            self.assertIn(
                "Load a model first", str(self.app.query_one(".error-text", Static).render())
            )

    async def test_streaming_leaves_navigation_responsive_and_rejects_duplicate_send(self):
        async with self.app.run_test(size=(180, 48)) as pilot:
            await self.settle(pilot)
            await pilot.press("f4")
            await self.settle(pilot)
            self.backend.chat_gate = asyncio.Event()
            editor = self.app.query_one("#chat-input", Composer)
            editor.load_text("A slow answer")
            self.app.query_one(ChatView).send_message()
            self.app.query_one(ChatView).send_message()
            await pilot.pause()
            self.assertTrue(editor.disabled)
            await pilot.press("f5")
            await pilot.pause()
            self.assertEqual(self.app.current_view, "logs")
            self.backend.chat_gate.set()
            await self.settle(pilot)
            self.assertEqual(sum(call[1] == "/api/chat/send" for call in self.backend.calls), 1)

    async def test_backend_outage_recovers_without_exiting(self):
        async with self.app.run_test(size=(100, 30)) as pilot:
            await self.settle(pilot)
            self.backend.online = False
            await self.app.poll_status()
            self.assertIn("unavailable", str(self.app.query_one("#connection", Static).render()))
            self.backend.online = True
            await self.app.poll_status()
            self.assertIn("Running", str(self.app.query_one("#connection", Static).render()))

    async def test_status_response_after_shutdown_does_not_touch_unmounted_widgets(self):
        async with self.app.run_test(size=(80, 24)) as pilot:
            await self.settle(pilot)
            self.backend.download_gate = asyncio.Event()
            pending = asyncio.create_task(self.app.poll_status())
            await asyncio.wait_for(self.backend.download_requested.wait(), timeout=2)
        self.assertFalse(self.app.is_running)
        self.backend.download_gate.set()
        await asyncio.wait_for(pending, timeout=2)
        self.assertFalse(self.app._polling)

    async def test_header_telemetry_refresh_resize_and_cpu_mode(self):
        async with self.app.run_test(size=(190, 52)) as pilot:
            await self.settle(pilot)
            self.assertTrue(self.app.query_one("#header-full").display)
            self.assertIn("Test GPU", str(self.app.query_one("#gpu-name", Static).render()))
            self.assertIn("6.0 / 24.0 GiB (25%)", str(self.app.query_one("#memory-value").render()))
            self.assertIn("Running / GPU", str(self.app.query_one("#connection").render()))
            self.assertIn("Tiny Q4_K_M", str(self.app.query_one("#active-model").render()))
            bar = self.app.query_one("#memory-bar")
            self.assertGreater(bar.region.height, 0)
            self.assertLessEqual(
                bar.region.bottom, self.app.query_one("#telemetry").content_region.bottom
            )
            self.backend.gpu["used_vram"] = 12
            self.backend.status["current_params"]["cpu_mode"] = True
            await self.app.poll_status()
            self.assertIn(
                "12.0 / 24.0 GiB (50%)", str(self.app.query_one("#memory-value").render())
            )
            self.assertIn("Running / CPU", str(self.app.query_one("#connection").render()))
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            self.assertFalse(self.app.query_one("#header-full").display)
            summary = str(self.app.query_one("#header-summary").render())
            for label in ("Tiny Q4_K_M", "Test GPU", "50%", "Running / CPU"):
                self.assertIn(label, summary)
            self.assertEqual(self.app.query_one("#masthead").region.height, 3)
            await pilot.resize_terminal(190, 52)
            await pilot.pause()
            self.assertTrue(self.app.query_one("#header-full").display)

    async def test_header_loading_stopped_offline_and_unavailable_telemetry(self):
        async with self.app.run_test(size=(190, 52)) as pilot:
            await self.settle(pilot)
            self.backend.gpu = {"name": "Apple M3", "vram": 32, "memory_kind": "unified"}
            await self.app.poll_status()
            self.assertIn("UNIFIED", str(self.app.query_one("#memory-label").render()))
            self.assertIn("usage N/A", str(self.app.query_one("#memory-value").render()))
            self.backend.gpu_available = False
            await self.app.poll_status()
            self.assertIn("Running", str(self.app.query_one("#connection").render()))
            self.assertIn("Unavailable", str(self.app.query_one("#memory-value").render()))
            self.backend.status["is_loading"] = True
            await self.app.poll_status()
            self.assertIn("Loading", str(self.app.query_one("#connection").render()))
            self.backend.status.update(is_loading=False, running=False)
            await self.app.poll_status()
            self.assertIn("Stopped", str(self.app.query_one("#connection").render()))
            self.assertIn("No model loaded", str(self.app.query_one("#active-model").render()))
            self.backend.online = False
            await self.app.poll_status()
            self.assertIn("unavailable", str(self.app.query_one("#connection").render()))
            self.assertIn("Unavailable", str(self.app.query_one("#active-model").render()))

    async def test_resize_preserves_model_selection_and_unsaved_settings(self):
        self.backend.models.append({
            **self.backend.models[0],
            "path": "/models/second.gguf",
            "name": "Second",
        })
        async with self.app.run_test(size=(180, 48)) as pilot:
            await self.settle(pilot)
            await pilot.press("f3")
            await self.settle(pilot)
            self.app.query_one("#local-table").focus()
            await pilot.press("down")
            await pilot.pause()
            self.app.query_one("#model-context", Input).value = "65536"
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            self.assertEqual(self.app.query_one("#models").selected_path, "/models/second.gguf")
            self.assertEqual(self.app.query_one("#model-context", Input).value, "65536")
            table = self.app.query_one("#local-table", DataTable)
            self.assertLessEqual(
                sum(column.width + 2 for column in table.columns.values()), table.size.width
            )

    async def test_rename_delete_and_refresh_saved_chat(self):
        async with self.app.run_test(size=(180, 48)) as pilot:
            await self.settle(pilot)
            await pilot.press("f4")
            await self.settle(pilot)
            await pilot.click("#chat-rename")
            await pilot.pause()
            self.app.screen.query_one(Input).value = "Renamed chat"
            await pilot.click("#save")
            await self.settle(pilot)
            self.assertEqual(self.backend.conversations["saved"]["title"], "Renamed chat")
            self.backend.conversations["saved"]["messages"].append({
                "role": "assistant",
                "content": "Newly saved elsewhere",
            })
            await pilot.press("ctrl+r")
            await self.settle(pilot)
            self.assertIn(
                "Newly saved elsewhere", [widget.source for widget in self.app.query(Markdown)]
            )
            await pilot.click("#chat-delete")
            await pilot.pause()
            await pilot.click("#confirm")
            await self.settle(pilot)
            self.assertNotIn("saved", self.backend.conversations)
            self.assertEqual(self.app.query_one(ChatView).conversation_id, "")

    async def test_quit_during_stream_closes_only_the_interface(self):
        async with self.app.run_test(size=(180, 48)) as pilot:
            await self.settle(pilot)
            await pilot.press("f4")
            await self.settle(pilot)
            self.backend.chat_gate = asyncio.Event()
            self.app.query_one("#chat-input", Composer).load_text("Pending response")
            self.app.query_one(ChatView).send_message()
            await pilot.pause()
            await pilot.press("ctrl+q")
            await pilot.pause()
            self.assertIsInstance(self.app.screen, Confirm)
            await pilot.click("#confirm")
        self.assertTrue(self.backend.status["running"])
        self.assertFalse(any(call[1] == "/api/models/eject" for call in self.backend.calls))
