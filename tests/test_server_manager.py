import os
import sys
import unittest
from unittest.mock import Mock, patch

from jinja2 import Template

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

    @patch("app.config.resolve_llama_server_bin", return_value="/usr/local/bin/llama-server")
    def test_build_command_includes_reasoning_budget(self, _mock_resolve):
        server = ServerManager()

        with patch(
            "app.server_manager.config_loader.get_llama_defaults",
            return_value=self.llama_defaults(),
        ):
            cmd = server._build_command(
                "/models/test-qwen.gguf",
                {"reasoning_budget": 2048, "reasoning_budget_message": "Act now."},
            )
            default_cmd = server._build_command("/models/test-qwen.gguf", {})

        self.assertEqual(cmd[cmd.index("--reasoning-budget") + 1], "2048")
        self.assertEqual(cmd[cmd.index("--reasoning-budget-message") + 1], "Act now.")
        self.assertNotIn("--reasoning-budget", default_cmd)

    @patch("app.config.resolve_llama_server_bin", return_value="/usr/local/bin/llama-server")
    def test_build_command_ignores_invalid_reasoning_budget(self, _mock_resolve):
        """A malformed profile value is skipped, not fatal to loading the model."""
        server = ServerManager()

        with patch(
            "app.server_manager.config_loader.get_llama_defaults",
            return_value=self.llama_defaults(),
        ):
            cmd = server._build_command(
                "/models/test-qwen.gguf",
                {"reasoning_budget": "none", "reasoning_budget_message": "Act now."},
            )
            negative_cmd = server._build_command("/models/test-qwen.gguf", {"reasoning_budget": -1})

        self.assertNotIn("--reasoning-budget", cmd)
        self.assertNotIn("--reasoning-budget-message", cmd)
        self.assertNotIn("--reasoning-budget", negative_cmd)

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

    @patch("app.config.resolve_llama_server_bin", return_value="/usr/local/bin/llama-server")
    def test_chatml_template_uses_chatml_turn_markers(self, _mock_resolve):
        """Guards against the ChatML template being clobbered (see PR #9 regression)."""
        server = ServerManager()

        with patch(
            "app.server_manager.config_loader.get_llama_defaults",
            return_value=self.llama_defaults(),
        ):
            cmd = server._build_command("/models/test-model.gguf", {"chat_template": "chatml"})

        template = cmd[cmd.index("--chat-template") + 1]
        self.assertIn("<|im_start|>", template)
        self.assertIn("<|im_end|>", template)
        self.assertNotIn("</think>", template)

    @patch("app.config.resolve_llama_server_bin", return_value="/usr/local/bin/llama-server")
    def test_chatml_template_renders_expected_prompt(self, _mock_resolve):
        server = ServerManager()

        with patch(
            "app.server_manager.config_loader.get_llama_defaults",
            return_value=self.llama_defaults(),
        ):
            cmd = server._build_command("/models/test-model.gguf", {"chat_template": "chatml"})

        template = cmd[cmd.index("--chat-template") + 1]
        rendered = Template(template).render(
            messages=[
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ],
            add_generation_prompt=True,
        )

        self.assertEqual(
            rendered,
            "<|im_start|>system\nbe terse<|im_end|>\n"
            "<|im_start|>user\nhi<|im_end|>\n"
            "<|im_start|>assistant\n",
        )


if __name__ == "__main__":
    unittest.main()
