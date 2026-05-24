import json
import os
import time
from pathlib import Path

import pytest

# Skip these tests in CI or standard pytest runs unless RUN_LOCAL_ONLY=1 is set
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LOCAL_ONLY") != "1",
    reason="Local-only integration tests. Run via tests/test_all.sh",
)

# Resolve workspace root and files
WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()
MODELS_JSON_PATH = Path(__file__).parent / "local_test_models.json"


def load_test_models() -> list[str]:
    """Load the list of target models to test from the editable JSON file."""
    if MODELS_JSON_PATH.exists():
        with open(MODELS_JSON_PATH) as f:
            return json.load(f)
    return [
        "Qwen3.6-35B-A3B-UD-Q5_K_M",
        "gemma-4-26B-A4B-it-Q8_0",
        "Qwopus3.6-27B-v2-MTP-Q4_K_S",
        "DeepSeek-R1-Distill-Qwen-32B-Q5_K_M",
    ]


@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    """Ensure we start with a clean state before running any model tests."""
    from app.server_manager import ServerManager

    server = ServerManager()
    if server.is_running:
        server.eject_model()
    yield
    if server.is_running:
        server.eject_model()


@pytest.mark.parametrize("model_query", load_test_models())
def test_model_tool_calling(model_query):
    """Load a model, query it to call the write_file tool, and verify file creation."""
    from app.chat import ChatManager
    from app.model_manager import scan_models
    from app.server_manager import ServerManager

    # 1. Locate the model GGUF file
    scanned_models = scan_models()
    resolved_path = None
    model_name = None

    for m in scanned_models:
        if m.name == model_query or model_query.lower() in m.name.lower():
            resolved_path = m.path
            model_name = m.name
            break

    if not resolved_path:
        pytest.skip(
            f"Model '{model_query}' not found locally in scanned directories. Skipping test."
        )

    print(f"\n--- Testing model: {model_name} ---")

    server = ServerManager()
    chat = ChatManager()

    # Create a fresh conversation
    chat.new_conversation()

    # Determine custom load settings for reasoning models
    params = {}
    if "deepseek" in model_name.lower():
        params["chat_template"] = "deepseek-r1"

    # 2. Boot/Load the model
    print(f"Loading model from {resolved_path}...")
    success = server.load_model(resolved_path, params)
    assert success, f"Failed to load model {model_name}"
    assert server.is_running, f"llama-server not running after loading {model_name}"

    # 3. Formulate a prompt that strongly instructs tool invocation
    prompt = (
        f"You must use the 'write_file' tool to write a file named 'hello_{model_name}.txt' "
        f"containing the message 'Hello from {model_name}!'. "
        "Do not explain your action or do anything else; simply call the tool."
    )

    test_file_path = WORKSPACE_ROOT / f"hello_{model_name}.txt"
    if test_file_path.exists():
        test_file_path.unlink()

    # 4. Stream chat response (this executes the tool under the hood synchronously)
    print("Sending prompt and streaming response...")
    stream = chat.stream_chat(
        user_message=prompt,
        temperature=0.1,  # Low temperature for highly deterministic tool calling
        max_tokens=1000,
    )

    # Consume the generator to trigger all tool executions and intermediate turns
    _ = list(stream)

    # 5. Verify the tool call was made and the file exists in the workspace
    assert test_file_path.exists(), (
        f"Model {model_name} failed to call write_file tool and create the file."
    )

    # Check content of created file
    content = test_file_path.read_text(encoding="utf-8").strip()
    print(f"File content created: '{content}'")
    assert f"Hello from {model_name}" in content or "Hello" in content, (
        f"File content mismatch: {content}"
    )

    # 6. Cleanup file and eject model to release memory for subsequent tests
    test_file_path.unlink()
    print(f"Cleaning up and ejecting model {model_name} to free VRAM...")
    server.eject_model()
    time.sleep(1.0)  # Grace period for complete process termination
