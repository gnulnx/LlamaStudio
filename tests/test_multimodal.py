import asyncio
import base64
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.model_manager import find_mmproj, resolve_model_projector, scan_models
from app.server_manager import ServerManager
from app.tools import ToolResult, read_file


class TestMultimodalFiles(unittest.TestCase):
    def test_read_file_returns_image_as_structured_multimodal_result(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
            workspace = Path(tmp_dir)
            image_bytes = b"\x89PNG\r\n\x1a\nsmall-image"
            image_path = workspace / "small.png"
            image_path.write_bytes(image_bytes)

            with (
                patch("app.tools.config_loader.sandbox_disabled", return_value=False),
                patch(
                    "app.tools.config_loader.get_workspace_root",
                    return_value=str(workspace),
                ),
            ):
                result = read_file("small.png")

        self.assertIsInstance(result, ToolResult)
        self.assertLess(len(result.content), 100)
        self.assertEqual(result.images[0]["mime_type"], "image/png")
        encoded = result.images[0]["data_url"].split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded), image_bytes)

    def test_read_file_keeps_text_behavior(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
            workspace = Path(tmp_dir)
            text_path = workspace / "notes.txt"
            text_path.write_text("hello", encoding="utf-8")

            with (
                patch("app.tools.config_loader.sandbox_disabled", return_value=False),
                patch(
                    "app.tools.config_loader.get_workspace_root",
                    return_value=str(workspace),
                ),
            ):
                result = read_file("notes.txt")

        self.assertEqual(result, "hello")


class TestMultimodalModelLoading(unittest.TestCase):
    def llama_defaults(self):
        return {
            "ctx_size": 16384,
            "gpu_layers": 999,
            "flash_attn": "on",
            "kv_cache_type": "q8_0",
            "vocab_type": "q8_0",
            "task_timeout": 900,
        }

    def test_scan_models_associates_sibling_projector(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
            model_dir = Path(tmp_dir)
            model_path = model_dir / "gemma-vision-Q4_K_M.gguf"
            projector_path = model_dir / "mmproj-gemma-f16.gguf"
            model_path.write_bytes(b"model")
            projector_path.write_bytes(b"projector")

            with patch(
                "app.model_manager.config_loader.get_model_directories",
                return_value=[str(model_dir)],
            ):
                models = scan_models()
                discovered_projector = find_mmproj(model_path)

        self.assertEqual(len(models), 1)
        self.assertTrue(models[0].is_multimodal)
        self.assertEqual(models[0].mmproj_path, str(projector_path))
        self.assertEqual(discovered_projector, str(projector_path))

    def test_resolve_model_projector_finds_downloadable_hub_artifact(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
            model_root = Path(tmp_dir)
            model_path = model_root / "HauhauCS" / "Gemma-4" / "model.gguf"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"model")

            details = {
                "siblings": [
                    {"rfilename": "model.gguf", "size": 4},
                    {"rfilename": "mmproj-Gemma-4-f16.gguf", "size": 990_000_000},
                ]
            }
            with (
                patch(
                    "app.model_manager.config_loader.get_model_directories",
                    return_value=[str(model_root)],
                ),
                patch(
                    "app.model_manager.get_huggingface_model_details",
                    new=AsyncMock(return_value=details),
                ),
            ):
                recovery = asyncio.run(resolve_model_projector(model_path))

        self.assertEqual(recovery["status"], "downloadable")
        self.assertEqual(recovery["repo_id"], "HauhauCS/Gemma-4")
        self.assertEqual(recovery["filename"], "mmproj-Gemma-4-f16.gguf")

    @patch("app.config.resolve_llama_server_bin", return_value="/usr/local/bin/llama-server")
    def test_build_command_includes_explicit_projector(self, _mock_resolve):
        server = ServerManager()
        with patch(
            "app.server_manager.config_loader.get_llama_defaults",
            return_value=self.llama_defaults(),
        ):
            command = server._build_command(
                "/models/gemma.gguf",
                {"mmproj": "/models/mmproj-gemma-f16.gguf"},
            )

        mmproj_index = command.index("--mmproj")
        self.assertEqual(command[mmproj_index + 1], "/models/mmproj-gemma-f16.gguf")


if __name__ == "__main__":
    unittest.main()
