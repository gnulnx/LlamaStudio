"""
Chat state management and llama.cpp API interaction.
"""
from __future__ import annotations
import httpx
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Generator
from .config import settings
from .logger import logger

@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class Conversation:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    title: str = "New Chat"

class ChatManager:
    _instance = None
    _conversations: dict[str, Conversation] = {}
    _active_id: str = ""

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_from_disk()
        return cls._instance

    def _load_from_disk(self):
        try:
            from pathlib import Path
            path = Path(settings.CONVERSATIONS_FILE)
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)
                    self._active_id = data.get("active_id", "")
                    self._conversations = {}
                    for c_id, c_data in data.get("conversations", {}).items():
                        messages = [
                            Message(
                                role=m["role"],
                                content=m["content"],
                                timestamp=m.get("timestamp", time.time())
                            )
                            for m in c_data.get("messages", [])
                        ]
                        self._conversations[c_id] = Conversation(
                            id=c_id,
                            messages=messages,
                            created_at=c_data.get("created_at", time.time()),
                            title=c_data.get("title", "New Chat")
                        )
            else:
                self._conversations = {}
                self._active_id = ""
        except Exception as e:
            logger.error(f"[chat] Error loading conversations: {e}")
            self._conversations = {}
            self._active_id = ""

    def _save_to_disk(self):
        try:
            from pathlib import Path
            path = Path(settings.CONVERSATIONS_FILE)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "active_id": self._active_id,
                "conversations": {
                    c_id: {
                        "id": conv.id,
                        "created_at": conv.created_at,
                        "title": conv.title,
                        "messages": [
                            {
                                "role": m.role,
                                "content": m.content,
                                "timestamp": m.timestamp
                            }
                            for m in conv.messages
                        ]
                    }
                    for c_id, conv in self._conversations.items()
                }
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[chat] Error saving conversations: {e}")

    def new_conversation(self) -> Conversation:
        conv = Conversation()
        self._conversations[conv.id] = conv
        self._active_id = conv.id
        self._save_to_disk()
        return conv

    def get_active(self) -> Conversation | None:
        if self._active_id not in self._conversations:
            return self.new_conversation()
        return self._conversations[self._active_id]

    def switch_to(self, conv_id: str) -> Conversation | None:
        if conv_id in self._conversations:
            self._active_id = conv_id
            self._save_to_disk()
            return self._conversations[conv_id]
        return None

    def rename_conversation(self, conv_id: str, new_title: str) -> bool:
        if conv_id in self._conversations:
            self._conversations[conv_id].title = new_title
            self._save_to_disk()
            return True
        return False

    def list_conversations(self) -> list[dict]:
        result = []
        for conv in self._conversations.values():
            # Auto-title from first user message if title is default "New Chat"
            title = conv.title
            if title == "New Chat":
                for msg in conv.messages:
                    if msg.role == "user":
                        title = msg.content[:50]
                        break
            result.append({
                "id": conv.id,
                "title": title,
                "message_count": len(conv.messages),
                "created_at": conv.created_at,
                "is_active": conv.id == self._active_id,
            })
        return result

    def delete_conversation(self, conv_id: str) -> bool:
        if conv_id in self._conversations:
            del self._conversations[conv_id]
            if self._active_id == conv_id:
                if self._conversations:
                    self._active_id = next(iter(self._conversations))
                else:
                    self.new_conversation()
            self._save_to_disk()
            return True
        return False

    def stream_chat(self, user_message: str, temperature: float = None,
                    top_p: float = None, max_tokens: int = None,
                    system_prompt: str = None) -> Generator[str, None, None]:
        """Send a message to llama-server and stream the response back."""
        conv = self.get_active()
        if conv is None:
            yield "Error: No active conversation"
            return

        # Add user message
        conv.messages.append(Message(role="user", content=user_message))
        self._save_to_disk()

        # Build message history for the API
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in conv.messages:
            messages.append({"role": msg.role, "content": msg.content})

        # Build the completion payload for llama.cpp server
        payload = {
            "messages": messages,
            "stream": True,
            "temperature": temperature or settings.DEFAULT_TEMPERATURE,
            "top_p": top_p or settings.DEFAULT_TOP_P,
            "max_tokens": max_tokens or settings.DEFAULT_MAX_TOKENS,
        }

        assistant_text = ""
        reasoning_text = ""
        try:
            with httpx.Client(timeout=120.0) as client:
                url = f"http://127.0.0.1:{settings.LLAMA_SERVER_PORT}/chat/completions"
                with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code != 200:
                        error_text = f"Server error {resp.status_code}: {resp.read().decode()}"
                        yield f"data: {json.dumps({'error': error_text})}\n\n"
                        return

                    # Send start marker
                    yield f"data: {json.dumps({'type': 'start'})}\n\n"

                    # Parse SSE stream
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]  # strip "data: "
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                # Handle both regular content and reasoning content
                                content = delta.get("content", "")
                                reasoning = delta.get("reasoning_content", "")
                                if reasoning:
                                    reasoning_text += reasoning
                                    yield f"data: {json.dumps({'reasoning': reasoning})}\n\n"
                                if content:
                                    assistant_text += content
                                    yield f"data: {json.dumps({'content': content})}\n\n"
                        except json.JSONDecodeError:
                            continue

                # Add assistant message to history
                # If no regular content was produced, use reasoning as the content
                final_text = assistant_text or reasoning_text
                conv.messages.append(Message(role="assistant", content=final_text))
                self._save_to_disk()

                # Send end marker
                yield f"data: {json.dumps({'type': 'end'})}\n\n"

        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': 'Cannot connect to llama-server'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

chat = ChatManager()
