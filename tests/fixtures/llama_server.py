"""Small llama-server protocol fixture; it checks real launch arguments and image input.

This replaces the external inference engine in browser integration tests. The
application still starts/stops a real process and talks to it over real HTTP.
It does not pretend to test model quality or GPU inference.
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    if "--version" in sys.argv:
        print("llama-server fixture version 1")
        return
    if "--help" in sys.argv:
        print("--mmap --mmproj --port --ctx-size --gpu-layers")
        return
    if "--list-devices" in sys.argv:
        print("CPU protocol fixture")
        return
    args = sys.argv[1:]
    projector = Path(args[args.index("--mmproj") + 1]) if "--mmproj" in args else None
    if projector is not None and not projector.read_bytes().startswith(b"GGUF"):
        raise ValueError("The projector passed to the engine is missing or invalid")
    events = Path(os.environ["LLS_TEST_ENGINE_EVENTS"])

    def record(data):
        with events.open("a") as output:
            output.write(json.dumps(data) + "\n")

    record({"type": "launch", "args": args, "pid": os.getpid()})
    startup = Path(os.environ["LLS_TEST_STARTUP_RELEASE"])
    deadline = time.monotonic() + 15
    while not startup.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not startup.exists():
        raise TimeoutError("Test did not release engine startup")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            data = {"status": "ok"}
            if self.path == "/v1/models":
                data = {
                    "data": [{"id": "fixture", "capabilities": ["vision"] if projector else []}]
                }
            elif self.path == "/props":
                data = {"modalities": {"vision": bool(projector), "audio": False}}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            record({"type": "completion", "payload": data})
            images = [
                part
                for message in data["messages"]
                if isinstance(message.get("content"), list)
                for part in message["content"]
                if part.get("type") == "image_url"
            ]
            if not projector or not images:
                self.send_error(
                    400, "Vision retry requires an attached projector and original image"
                )
                return
            release = Path(os.environ["LLS_TEST_INFERENCE_RELEASE"])
            deadline = time.monotonic() + 15
            while not release.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            if not release.exists():
                self.send_error(503, "Test did not release the inference request")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {"choices": [{"delta": {"content": "IMAGE_RECOVERY_OK"}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())

    ThreadingHTTPServer(("127.0.0.1", int(args[args.index("--port") + 1])), Handler).serve_forever()


if __name__ == "__main__":
    main()
