"""Exercise text tool calls through the real chat parser and filesystem tool.

The previous test implemented a separate parser here and only tested its copy.
Only the external model stream is supplied by this fixture.
"""

import json

import httpx
import pytest

from app.chat import chat


@pytest.mark.parametrize("syntax", ["gemma", "deepseek", "deepseek_fullwidth"])
def test_text_tool_call_executes_write_and_returns_result_to_model(tmp_path, monkeypatch, syntax):
    expected = "The real chat pipeline wrote this file."
    arguments = {"file_path": "hello.txt", "content": expected}
    if syntax == "gemma":
        tool_text = (
            f"<|tool_call>call:write_file(filename='hello.txt', content={expected!r})<tool_call|>"
        )
    else:
        tool_text = (
            "<|tool_calls_begin|><|tool_call_begin|>function<|tool_sep|>write_file\n```json\n"
            + json.dumps(arguments)
            + "\n```<|tool_call_end|><|tool_calls_end|>"
        )
        if syntax == "deepseek_fullwidth":
            tool_text = (
                tool_text
                .replace("|", "\uff5c")
                .replace("tool_", "tool\u2581")
                .replace("calls_", "calls\u2581")
                .replace("call_", "call\u2581")
            )
    requests = []

    def model_response(request):
        payload = json.loads(request.content)
        requests.append(payload)
        content = tool_text if len(requests) == 1 else "File created successfully."
        chunk = {"choices": [{"delta": {"content": content}}]}
        return httpx.Response(
            200,
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            headers={"Content-Type": "text/event-stream"},
        )

    client = httpx.Client
    transport = httpx.MockTransport(model_response)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=transport, **kwargs))
    monkeypatch.setattr(chat, "_conversations", {})
    monkeypatch.setattr(chat, "_active_id", "")
    monkeypatch.setattr("app.chat.settings.CONVERSATIONS_FILE", str(tmp_path / "conversation.json"))
    events = list(chat.stream_chat("Write hello.txt", workspace_root=str(tmp_path)))
    assert (tmp_path / "hello.txt").read_text() == expected
    assert len(requests) == 2
    results = [message for message in requests[1]["messages"] if message["role"] == "tool"]
    assert len(results) == 1
    assert "Successfully wrote" in results[0]["content"]
    assert any('"type": "tool_exec_end"' in event for event in events)
    assert chat.get_active().messages[-1].content == "File created successfully."
    saved = json.loads((tmp_path / "conversation.json").read_text())
    assert (
        saved["conversations"][chat.get_active().id]["messages"][-1]["content"]
        == "File created successfully."
    )
