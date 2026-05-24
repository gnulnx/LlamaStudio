import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.chat import chat


class FakeStreamResponse:
    status_code = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_lines(self):
        chunk = {
            "choices": [
                {
                    "delta": {
                        "content": "Plain response with no tool call.",
                    }
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}"
        yield "data: [DONE]"


class FakeHttpClient:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, json):
        return FakeStreamResponse()


class TestChatStreaming(unittest.TestCase):
    def test_plain_response_stream_does_not_shadow_regex_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            conversation_path = str(Path(tmp) / "conversations.json")
            chat._conversations = {}
            chat._active_id = ""

            with (
                patch("app.chat.settings.CONVERSATIONS_FILE", conversation_path),
                patch("app.chat.httpx.Client", FakeHttpClient),
            ):
                events = list(chat.stream_chat("hello"))

        self.assertTrue(any('"type": "end"' in event for event in events))
        self.assertTrue(any("Plain response with no tool call." in event for event in events))


if __name__ == "__main__":
    unittest.main()
