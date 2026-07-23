import os
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.server_manager import ServerManager


class TestServerManagerCommand(unittest.TestCase):
    def llama_defaults(self):
        return {
            "ctx_size": 16384,
            "gpu_layers": 999,
            "flash_attn": "on",
            "kv_cache_type": "q8_0",
            "vocab_type": "q8_0",
            "task_timeout": 900,
        }

    @patch("app.config.resolve_llama_server_bin", return_value="/usr/local/bin/llama-server")
    def test_build_command_includes_task_timeout(self, _mock_resolve):
        server = ServerManager()

        with patch(
            "app.server_manager.config_loader.get_llama_defaults",
            return_value=self.llama_defaults(),
        ):
            cmd = server._build_command("/models/test-qwen.gguf", {})

        self.assertIn("--timeout", cmd)
        timeout_index = cmd.index("--timeout")
        self.assertEqual(cmd[timeout_index + 1], "900")

    def test_supports_audio_reads_active_server_modalities(self):
        server = ServerManager()
        response = Mock()
        response.json.return_value = {"modalities": {"vision": True, "audio": True}}

        with (
            patch.object(server, "_port_in_use", return_value=True),
            patch("httpx.get", return_value=response),
        ):
            self.assertTrue(server.supports_audio())

        response.raise_for_status.assert_called_once()

    @patch("app.config.resolve_llama_server_bin", return_value="/usr/local/bin/llama-server")
    def test_build_command_allows_task_timeout_override(self, _mock_resolve):
        server = ServerManager()
        with patch(
            "app.server_manager.config_loader.get_llama_defaults",
            return_value=self.llama_defaults(),
        ):
            cmd = server._build_command("/models/test-qwen.gguf", {"task_timeout": 1200})

        self.assertIn("--timeout", cmd)
        timeout_index = cmd.index("--timeout")
        self.assertEqual(cmd[timeout_index + 1], "1200")


if __name__ == "__main__":
    unittest.main()
