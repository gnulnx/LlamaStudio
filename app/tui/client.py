"""Asynchronous HTTP/SSE transport; the backend owns models and persistence."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx


class APIError(Exception):
    """An error suitable for displaying to the operator."""


def error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        if "detail" in payload:
            return error_message(payload["detail"])
        return "\n".join(
            str(payload[k]) for k in ("title", "message", "hint") if payload.get(k)
        ) or str(payload)
    return str(payload)


def decode_event(data: str) -> dict[str, Any] | None:
    # Older servers emit Python-style quotes on the two wrapper events only.
    # Never evaluate arbitrary stream content.
    if data in ("{'type': 'start'}", "{'type': 'end'}"):
        return {"type": "start" if "start" in data else "end"}
    if data == "[DONE]":
        return {"type": "end"}
    try:
        event = json.loads(data)
    except ValueError as exc:
        raise APIError("The server sent an invalid chat event. Please retry.") from exc
    return event if isinstance(event, dict) else None


class StudioClient:
    def __init__(self, base_url: str, *, transport: httpx.AsyncBaseTransport | None = None):
        self.http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(30, connect=5),
            transport=transport,
        )

    async def close(self) -> None:
        await self.http.aclose()

    @staticmethod
    def check_response(response: httpx.Response) -> None:
        if response.is_error:
            try:
                detail = error_message(response.json())
            except ValueError:
                detail = response.text[:1000]
            raise APIError(detail or f"Server returned HTTP {response.status_code}.")

    async def request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = await self.http.request(method, path, **kwargs)
            self.check_response(response)
            return response.json()
        except httpx.TimeoutException as exc:
            raise APIError("The server timed out. Check Logs and refresh before retrying.") from exc
        except httpx.RequestError as exc:
            raise APIError("Cannot reach LlamaStudio. Check the backend, then refresh.") from exc
        except ValueError as exc:
            raise APIError("The server returned an invalid response.") from exc

    async def get(self, path: str, **params: Any) -> Any:
        return await self.request("GET", path, params=params)

    async def post(self, endpoint: str, **payload: Any) -> Any:
        return await self.request(
            "POST", endpoint, json=payload, timeout=httpx.Timeout(180, connect=5)
        )

    async def chat(self, **payload: Any) -> AsyncIterator[dict[str, Any]]:
        """Read complete SSE frames, including frames split across network chunks."""
        try:
            async with self.http.stream(
                "POST",
                "/api/chat/send",
                json=payload,
                timeout=httpx.Timeout(900, connect=5),
            ) as response:
                if response.is_error:
                    await response.aread()
                    self.check_response(response)
                lines: list[str] = []
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        lines.append(line[5:].lstrip(" "))
                    elif not line and lines:
                        event = decode_event("\n".join(lines))
                        lines.clear()
                        if event is not None:
                            yield event
                            if event.get("type") == "end":
                                return
                if lines:
                    event = decode_event("\n".join(lines))
                    if event is not None:
                        yield event
                        if event.get("type") == "end":
                            return
                raise APIError("The chat stream disconnected before completion.")
        except httpx.TimeoutException as exc:
            raise APIError("The model stopped responding. Check Logs before retrying.") from exc
        except httpx.RequestError as exc:
            raise APIError(
                "The chat connection was lost. Your saved conversation remains on the server."
            ) from exc
