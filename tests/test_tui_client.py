"""Transport contract tests without an inference server or network access."""

import unittest

import httpx

from app.tui.client import APIError, StudioClient


class Chunks(httpx.AsyncByteStream):
    async def __aiter__(self):
        for chunk in (
            b": keepalive\r\ndata: {'type': 'start'}\r\n\r\nda",
            b'ta: {"reasoning":"think"}\n\ndata: {"content":',
            b'"hello"}\n\ndata: {"type":"tool_exec_start","name":"read_file"}\n\n',
            b'data: {"type":"tool_exec_end","result":"ok"}\n\n',
            b"data: [DONE]\n\n",
        ):
            yield chunk


class TestStudioClient(unittest.IsolatedAsyncioTestCase):
    async def test_fragmented_stream_and_legacy_wrappers(self):
        client = StudioClient(
            "http://studio",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Chunks())),
        )
        try:
            events = [event async for event in client.chat(message="hello")]
            self.assertEqual(events[1], {"reasoning": "think"})
            self.assertEqual(events[2], {"content": "hello"})
            self.assertEqual(events[3]["type"], "tool_exec_start")
            self.assertEqual(events[4]["result"], "ok")
            self.assertEqual(events[-1], {"type": "end"})
        finally:
            await client.close()

    async def test_http_error_is_actionable_for_requests_and_streams(self):
        client = StudioClient(
            "http://studio",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(503, json={"detail": "Load a model first"})
            ),
        )
        try:
            with self.assertRaisesRegex(APIError, "Load a model first"):
                await client.get("/api/models")
            with self.assertRaisesRegex(APIError, "Load a model first"):
                _ = [event async for event in client.chat(message="hello")]
        finally:
            await client.close()

    async def test_truncated_stream_is_not_success(self):
        client = StudioClient(
            "http://studio",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text='data: {"content":"partial"}\n\n')
            ),
        )
        try:
            with self.assertRaisesRegex(APIError, "disconnected"):
                _ = [event async for event in client.chat(message="hello")]
        finally:
            await client.close()

    async def test_connection_failure_is_readable(self):
        def offline(request):
            raise httpx.ConnectError("connection refused", request=request)

        client = StudioClient("http://studio", transport=httpx.MockTransport(offline))
        try:
            with self.assertRaisesRegex(APIError, "Cannot reach"):
                await client.get("/api/server/status")
        finally:
            await client.close()
