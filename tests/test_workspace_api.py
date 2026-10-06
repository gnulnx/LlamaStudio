from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app.main import send_message


class TestChatWorkspaceAPI(unittest.TestCase):
    def send(self, body: dict):
        request = SimpleNamespace(json=AsyncMock(return_value=body))
        return asyncio.run(send_message(request))

    def test_invalid_workspace_values_return_bad_request(self):
        for value in (True, False, 42, 0, 3.14, {}, [], ["path"], "", " ", "\0"):
            with self.subTest(value=value), self.assertRaises(HTTPException) as error:
                self.send({"message": "hello", "workspace_root": value})
            self.assertEqual(error.exception.status_code, 400)

    def test_workspace_must_be_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "file.txt"
            file_path.touch()
            for value in (str(file_path), str(Path(tmp_dir) / "missing")):
                with self.subTest(value=value), self.assertRaises(HTTPException) as error:
                    self.send({"message": "hello", "workspace_root": value})
                self.assertEqual(error.exception.status_code, 400)

    def test_path_resolution_errors_return_bad_request(self):
        for failure in (OSError("unreadable path"), RuntimeError("symlink loop")):
            with (
                self.subTest(failure=failure),
                patch("app.main.Path.resolve", side_effect=failure),
                self.assertRaises(HTTPException) as error,
            ):
                self.send({"message": "hello", "workspace_root": "/workspace"})
            self.assertEqual(error.exception.status_code, 400)

    def test_canonical_workspace_is_forwarded_to_chat(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir).resolve()
            child = workspace / "child"
            child.mkdir()
            with (
                patch("app.main.server") as server,
                patch("app.main.chat.stream_chat", return_value=iter(())) as stream,
                patch("app.main.StreamingResponse", side_effect=lambda events, **kwargs: events),
            ):
                server.is_running = True
                events = self.send({"message": "hello", "workspace_root": str(child / "..")})
                list(events)
            self.assertEqual(stream.call_args.kwargs["workspace_root"], str(workspace))

    def test_absent_or_null_workspace_keeps_configured_default(self):
        for fields in ({}, {"workspace_root": None}):
            with (
                self.subTest(fields=fields),
                patch("app.main.server") as server,
                patch("app.main.chat.stream_chat", return_value=iter(())) as stream,
                patch("app.main.StreamingResponse", side_effect=lambda events, **kwargs: events),
            ):
                server.is_running = True
                list(self.send({"message": "hello", **fields}))
                self.assertIsNone(stream.call_args.kwargs["workspace_root"])
