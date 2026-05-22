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
    role: str  # "user", "assistant", "system", "tool"
    content: str | None = None
    timestamp: float = field(default_factory=time.time)
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

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
                                content=m.get("content"),
                                timestamp=m.get("timestamp", time.time()),
                                tool_calls=m.get("tool_calls"),
                                tool_call_id=m.get("tool_call_id"),
                                name=m.get("name")
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
                                "timestamp": m.timestamp,
                                "tool_calls": m.tool_calls,
                                "tool_call_id": m.tool_call_id,
                                "name": m.name
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

    def stream_chat(self, user_message: str | None = None, temperature: float = None,
                    top_p: float = None, max_tokens: int = None,
                    system_prompt: str = None) -> Generator[str, None, None]:
        """Send a message to llama-server and stream the response, automatically executing tools if requested."""
        from .tools import ALL_TOOLS, execute_tool

        conv = self.get_active()
        if conv is None:
            yield "Error: No active conversation"
            return

        # Add user message if provided
        if user_message is not None:
            conv.messages.append(Message(role="user", content=user_message))
            self._save_to_disk()

        while True:
            # Build message history for the API
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            for msg in conv.messages:
                msg_dict = {"role": msg.role, "content": msg.content}
                if msg.tool_calls:
                    msg_dict["tool_calls"] = msg.tool_calls
                if msg.tool_call_id:
                    msg_dict["tool_call_id"] = msg.tool_call_id
                if msg.name:
                    msg_dict["name"] = msg.name
                messages.append(msg_dict)

            # Build the completion payload for llama.cpp server
            payload = {
                "messages": messages,
                "stream": True,
                "temperature": temperature or settings.DEFAULT_TEMPERATURE,
                "top_p": top_p or settings.DEFAULT_TOP_P,
                "max_tokens": max_tokens or settings.DEFAULT_MAX_TOKENS,
                "tools": ALL_TOOLS,
                "tool_choice": "auto"
            }

            assistant_text = ""
            reasoning_text = ""
            tool_calls_accumulated = []

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
                                    
                                    # Handle tool calls
                                    tool_calls = delta.get("tool_calls", [])
                                    if tool_calls:
                                        for tc in tool_calls:
                                            idx = tc.get("index", 0)
                                            # Ensure our list is long enough
                                            while len(tool_calls_accumulated) <= idx:
                                                tool_calls_accumulated.append({
                                                    "id": "",
                                                    "type": "function",
                                                    "function": {"name": "", "arguments": ""}
                                                })
                                            
                                            accum = tool_calls_accumulated[idx]
                                            if tc.get("id"):
                                                accum["id"] = tc["id"]
                                            func = tc.get("function", {})
                                            if func.get("name"):
                                                accum["function"]["name"] = func["name"]
                                            if func.get("arguments"):
                                                accum["function"]["arguments"] += func["arguments"]
                                            
                                            # Stream raw tool calls to the frontend UI
                                            yield f"data: {json.dumps({'tool_call_delta': tc})}\n\n"

                            except json.JSONDecodeError:
                                continue

                # Post-processing assistant output
                if tool_calls_accumulated:
                    # Clean up and ensure every tool call has an ID
                    cleaned_tcs = []
                    for tc in tool_calls_accumulated:
                        cleaned_tcs.append({
                            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]
                            }
                        })
                    
                    # Save assistant message with tool calls in database
                    conv.messages.append(Message(
                        role="assistant",
                        content=assistant_text or None,
                        tool_calls=cleaned_tcs
                    ))
                    self._save_to_disk()

                    # Execute tools sequentially
                    for tc in cleaned_tcs:
                        tc_id = tc["id"]
                        name = tc["function"]["name"]
                        args_str = tc["function"]["arguments"]
                        
                        try:
                            args = json.loads(args_str) if args_str else {}
                        except Exception as e:
                            logger.error(f"Failed to parse tool arguments: {args_str}. Error: {e}")
                            args = {}

                        # Notify frontend that tool execution is starting
                        yield f"data: {json.dumps({'type': 'tool_exec_start', 'id': tc_id, 'name': name, 'arguments': args})}\n\n"
                        
                        # Execute the tool
                        logger.info(f"Executing tool '{name}' with arguments: {args}")
                        result = execute_tool(name, args)
                        
                        # Notify frontend that tool execution finished
                        yield f"data: {json.dumps({'type': 'tool_exec_end', 'id': tc_id, 'name': name, 'result': result})}\n\n"

                        # Save the tool output message in conversation
                        conv.messages.append(Message(
                            role="tool",
                            content=result,
                            tool_call_id=tc_id,
                            name=name
                        ))
                        self._save_to_disk()
                    
                    # Continue the while loop to get the next response from the model
                    continue
                else:
                    # Add standard assistant message to history
                    final_text = assistant_text or reasoning_text
                    conv.messages.append(Message(role="assistant", content=final_text))
                    self._save_to_disk()
                    
                    # Send end marker and break
                    yield f"data: {json.dumps({'type': 'end'})}\n\n"
                    break

            except httpx.ConnectError:
                yield f"data: {json.dumps({'error': 'Cannot connect to llama-server'})}\n\n"
                break
            except Exception as e:
                logger.error(f"Error in stream_chat: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break

chat = ChatManager()
