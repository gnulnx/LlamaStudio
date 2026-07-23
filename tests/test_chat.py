import base64
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import httpx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.chat import MAX_TOOL_RESULT_CHARS, AudioAttachment, ImageAttachment, chat
from app.tools import ToolResult


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
        usage_chunk = {
            "choices": [],
            "usage": {
                "prompt_tokens": 19,
                "completion_tokens": 7,
                "total_tokens": 26,
                "prompt_tokens_details": {"cached_tokens": 5},
            },
            "timings": {
                "prompt_ms": 42.254,
                "predicted_ms": 18.364,
                "predicted_per_second": 163.363,
            },
        }
        yield f"data: {json.dumps(usage_chunk)}"
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


class FakeToolStreamResponse:
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
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_list_dir",
                                "type": "function",
                                "function": {
                                    "name": "list_dir",
                                    "arguments": json.dumps({"dir_path": "."}),
                                },
                            }
                        ]
                    }
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}"
        yield "data: [DONE]"


class FakeReadImageToolStreamResponse(FakeToolStreamResponse):
    def iter_lines(self):
        chunk = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_read_image",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"file_path": "small.png"}),
                                },
                            }
                        ]
                    }
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}"
        yield "data: [DONE]"


class FakeToolHttpClient:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, json):
        return FakeToolStreamResponse()


class CapturingHttpClient(FakeHttpClient):
    payloads: ClassVar[list[dict]] = []

    def stream(self, method, url, json):
        self.payloads.append(json)
        return FakeStreamResponse()


class CapturingToolThenTextHttpClient(FakeHttpClient):
    payloads: ClassVar[list[dict]] = []

    def stream(self, method, url, json):
        self.payloads.append(json)
        if len(self.payloads) == 1:
            return FakeReadImageToolStreamResponse()
        return FakeStreamResponse()


class FakeReadTimeoutHttpClient:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, json):
        raise httpx.ReadTimeout("timed out")


class TestChatStreaming(unittest.TestCase):
    @staticmethod
    def image_attachment():
        image_bytes = b"\x89PNG\r\n\x1a\nsmall-image"
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return ImageAttachment.from_payload({
            "name": "small.png",
            "mime_type": "image/png",
            "data_url": f"data:image/png;base64,{encoded}",
            "size": len(image_bytes),
        })

    @staticmethod
    def audio_attachment():
        audio_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt "
        encoded = base64.b64encode(audio_bytes).decode("ascii")
        return AudioAttachment.from_payload({
            "name": "hello.wav",
            "mime_type": "audio/wav",
            "data_url": f"data:audio/wav;base64,{encoded}",
            "size": len(audio_bytes),
        })

    def chat_defaults(self, max_tool_iterations=50):
        return {
            "system_prompt": "You are a helpful assistant.",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048,
            "request_timeout": 900,
            "max_tool_iterations": max_tool_iterations,
        }

    def test_plain_response_stream_does_not_shadow_regex_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            conversation_path = str(Path(tmp) / "conversations.json")
            chat._conversations = {}
            chat._active_id = ""

            with (
                patch("app.chat.settings.CONVERSATIONS_FILE", conversation_path),
                patch(
                    "app.chat.config_loader.get_chat_defaults", return_value=self.chat_defaults()
                ),
                patch("app.chat.httpx.Client", FakeHttpClient),
            ):
                events = list(chat.stream_chat("hello"))
                conv = chat.get_active()

        self.assertTrue(any('"type": "end"' in event for event in events))
        self.assertTrue(any("Plain response with no tool call." in event for event in events))
        self.assertTrue(any('"type": "metrics"' in event for event in events))
        self.assertIsNotNone(conv)
        self.assertEqual(conv.messages[-1].metrics["total_tokens"], 26)
        self.assertEqual(conv.messages[-1].metrics["cached_tokens"], 5)
        self.assertEqual(conv.messages[-1].metrics["prompt_seconds"], 0.042)
        self.assertEqual(conv.messages[-1].metrics["generation_seconds"], 0.018)
        self.assertEqual(conv.messages[-1].metrics["tokens_per_second"], 163.363)

    def test_tool_iteration_limit_is_configurable_and_saved_as_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            conversation_path = str(Path(tmp) / "conversations.json")
            chat._conversations = {}
            chat._active_id = ""

            with (
                patch("app.chat.settings.CONVERSATIONS_FILE", conversation_path),
                patch(
                    "app.chat.config_loader.get_chat_defaults",
                    return_value=self.chat_defaults(max_tool_iterations=2),
                ),
                patch("app.chat.httpx.Client", FakeToolHttpClient),
                patch("app.tools.execute_tool", return_value="tool result"),
            ):
                events = list(chat.stream_chat("inspect files"))
                conv = chat.get_active()

        joined_events = "".join(events)
        self.assertIn("Stopped after 2 tool-calling rounds", joined_events)
        self.assertNotIn('"error"', joined_events)
        self.assertIsNotNone(conv)
        self.assertEqual(conv.messages[-1].role, "assistant")
        self.assertIn("Stopped after 2 tool-calling rounds", conv.messages[-1].content)

    def test_read_timeout_is_saved_as_assistant_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            conversation_path = str(Path(tmp) / "conversations.json")
            chat._conversations = {}
            chat._active_id = ""

            with (
                patch("app.chat.settings.CONVERSATIONS_FILE", conversation_path),
                patch(
                    "app.chat.config_loader.get_chat_defaults", return_value=self.chat_defaults()
                ),
                patch("app.chat.httpx.Client", FakeReadTimeoutHttpClient),
            ):
                events = list(chat.stream_chat("summarize a large file"))
                conv = chat.get_active()

        joined_events = "".join(events)
        self.assertIn("Timed out waiting for llama-server", joined_events)
        self.assertNotIn('"error"', joined_events)
        self.assertIsNotNone(conv)
        self.assertEqual(conv.messages[-1].role, "assistant")
        self.assertIn("Timed out waiting for llama-server", conv.messages[-1].content)

    def test_direct_image_is_sent_as_image_url_content_not_tokenized_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            conversation_path = str(Path(tmp) / "conversations.json")
            chat._conversations = {}
            chat._active_id = ""
            CapturingHttpClient.payloads = []

            with (
                patch("app.chat.settings.CONVERSATIONS_FILE", conversation_path),
                patch(
                    "app.chat.config_loader.get_chat_defaults",
                    return_value=self.chat_defaults(),
                ),
                patch("app.chat.httpx.Client", CapturingHttpClient),
                patch(
                    "app.server_manager.server.supports_multimodal",
                    return_value=True,
                ),
            ):
                list(
                    chat.stream_chat(
                        "What is in this image?",
                        images=[self.image_attachment()],
                        enable_thinking=False,
                    )
                )

        content = next(
            message["content"]
            for message in CapturingHttpClient.payloads[0]["messages"]
            if message["role"] == "user"
        )
        self.assertEqual(content[0], {"type": "text", "text": "What is in this image?"})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(CapturingHttpClient.payloads[0]["stream_options"], {"include_usage": True})
        self.assertEqual(
            CapturingHttpClient.payloads[0]["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_read_file_image_becomes_short_tool_text_plus_multimodal_user_part(self):
        with tempfile.TemporaryDirectory() as tmp:
            conversation_path = str(Path(tmp) / "conversations.json")
            chat._conversations = {}
            chat._active_id = ""
            CapturingToolThenTextHttpClient.payloads = []
            attachment = self.image_attachment()
            tool_result = ToolResult(
                content="Loaded image 'small.png' (19 bytes) as multimodal input.",
                images=[attachment.to_dict()],
            )

            with (
                patch("app.chat.settings.CONVERSATIONS_FILE", conversation_path),
                patch(
                    "app.chat.config_loader.get_chat_defaults",
                    return_value=self.chat_defaults(max_tool_iterations=3),
                ),
                patch("app.chat.httpx.Client", CapturingToolThenTextHttpClient),
                patch("app.tools.execute_tool", return_value=tool_result),
                patch(
                    "app.server_manager.server.supports_multimodal",
                    return_value=True,
                ),
            ):
                list(chat.stream_chat("Inspect small.png"))

        second_messages = CapturingToolThenTextHttpClient.payloads[1]["messages"]
        tool_message = next(message for message in second_messages if message["role"] == "tool")
        image_message = second_messages[second_messages.index(tool_message) + 1]
        self.assertLess(len(tool_message["content"]), 100)
        self.assertEqual(image_message["role"], "user")
        self.assertEqual(image_message["content"][1]["type"], "image_url")

    def test_direct_audio_is_sent_as_input_audio_content_not_tokenized_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            conversation_path = str(Path(tmp) / "conversations.json")
            chat._conversations = {}
            chat._active_id = ""
            CapturingHttpClient.payloads = []
            attachment = self.audio_attachment()

            with (
                patch("app.chat.settings.CONVERSATIONS_FILE", conversation_path),
                patch(
                    "app.chat.config_loader.get_chat_defaults",
                    return_value=self.chat_defaults(),
                ),
                patch("app.chat.httpx.Client", CapturingHttpClient),
                patch("app.server_manager.server.supports_audio", return_value=True),
            ):
                list(
                    chat.stream_chat(
                        "Transcribe this audio.",
                        audios=[attachment],
                        enable_thinking=False,
                    )
                )

        content = next(
            message["content"]
            for message in CapturingHttpClient.payloads[0]["messages"]
            if message["role"] == "user"
        )
        self.assertEqual(content[0]["type"], "input_audio")
        self.assertEqual(content[0]["input_audio"]["format"], "wav")
        self.assertEqual(content[1], {"type": "text", "text": "Transcribe this audio."})
        self.assertEqual(
            base64.b64decode(content[0]["input_audio"]["data"]),
            b"RIFF\x24\x00\x00\x00WAVEfmt ",
        )
        self.assertNotIn("tools", CapturingHttpClient.payloads[0])

    def test_read_file_audio_becomes_short_tool_text_plus_multimodal_user_part(self):
        with tempfile.TemporaryDirectory() as tmp:
            conversation_path = str(Path(tmp) / "conversations.json")
            chat._conversations = {}
            chat._active_id = ""
            CapturingToolThenTextHttpClient.payloads = []
            attachment = self.audio_attachment()
            tool_result = ToolResult(
                content="Loaded audio 'hello.wav' (16 bytes) as multimodal input.",
                audios=[attachment.to_dict()],
            )

            with (
                patch("app.chat.settings.CONVERSATIONS_FILE", conversation_path),
                patch(
                    "app.chat.config_loader.get_chat_defaults",
                    return_value=self.chat_defaults(max_tool_iterations=3),
                ),
                patch("app.chat.httpx.Client", CapturingToolThenTextHttpClient),
                patch("app.tools.execute_tool", return_value=tool_result),
                patch("app.server_manager.server.supports_audio", return_value=True),
            ):
                list(chat.stream_chat("Inspect hello.wav"))

        second_messages = CapturingToolThenTextHttpClient.payloads[1]["messages"]
        tool_message = next(message for message in second_messages if message["role"] == "tool")
        audio_message = second_messages[second_messages.index(tool_message) + 1]
        self.assertLess(len(tool_message["content"]), 100)
        self.assertEqual(audio_message["role"], "user")
        self.assertEqual(audio_message["content"][0]["type"], "input_audio")
        self.assertIn("tools", CapturingToolThenTextHttpClient.payloads[0])
        self.assertNotIn("tools", CapturingToolThenTextHttpClient.payloads[1])

    def test_legacy_oversized_tool_output_is_bounded_before_api_request(self):
        from app.chat import _bounded_tool_result

        bounded = _bounded_tool_result("x" * (MAX_TOOL_RESULT_CHARS + 500))

        self.assertIn("Tool output truncated", bounded)
        self.assertLess(len(bounded), MAX_TOOL_RESULT_CHARS + 100)


if __name__ == "__main__":
    unittest.main()
