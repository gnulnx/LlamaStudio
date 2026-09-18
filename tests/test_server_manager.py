import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
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


class TestStartupFailureReporting(unittest.TestCase):
    """A failed load should say what llama-server said, not guess at causes."""

    UNSUPPORTED_QUANT_LOG = """--- LLamaStudio Server Starting ---
Model Path: /models/Ternary-Bonsai-2-27B-PQ2_0.gguf

0.00.064.811 I srv    load_model: loading model '/models/Ternary-Bonsai-2-27B-PQ2_0.gguf'
0.00.095.171 E gguf_init_from_reader: tensor 'output.weight' has invalid ggml type 142. should be in [0, 43)
0.00.095.175 E gguf_init_from_reader: failed to read tensor info
0.00.097.356 E llama_model_load: error loading model: llama_model_loader: failed to load model
0.00.127.258 E srv  llama_server: exiting due to model loading error
"""

    OUT_OF_MEMORY_LOG = """--- LLamaStudio Server Starting ---

0.00.900.000 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 20480.00 MiB on device 0: cudaMalloc failed: out of memory
0.00.900.100 E llama_model_load: error loading model: unable to allocate CUDA0 buffer
"""

    def write_log(self, contents: str) -> Path:
        directory = tempfile.mkdtemp()
        log_file = Path(directory) / "server.log"
        log_file.write_text(contents)
        self.addCleanup(shutil.rmtree, directory)
        return log_file

    def test_reports_the_root_cause_not_the_last_consequence(self):
        """llama.cpp logs the specific fault first and the generic fallout after."""
        errors = ServerManager._startup_errors(self.write_log(self.UNSUPPORTED_QUANT_LOG))

        self.assertIn("invalid ggml type 142", errors[0])
        self.assertNotIn("DEBUG INFO", " ".join(errors))

    def test_ignores_decoration_written_by_a_previous_attempt(self):
        log_file = self.write_log(
            self.UNSUPPORTED_QUANT_LOG
            + "\nERROR: llama-server reported:\n- something from an earlier run\n"
        )
        errors = ServerManager._startup_errors(log_file)

        self.assertNotIn("earlier run", " ".join(errors))

    def test_skips_the_cpu_retry_for_a_file_llama_cpp_cannot_parse(self):
        """A quantisation the build has no type for fails identically on CPU."""
        server = ServerManager()
        errors = ServerManager._startup_errors(self.write_log(self.UNSUPPORTED_QUANT_LOG))

        self.assertTrue(server._retry_on_cpu_is_futile(errors))

    def test_still_retries_on_cpu_when_the_gpu_ran_out_of_memory(self):
        server = ServerManager()
        errors = ServerManager._startup_errors(self.write_log(self.OUT_OF_MEMORY_LOG))

        self.assertFalse(server._retry_on_cpu_is_futile(errors))

    def test_a_log_with_no_errors_yields_nothing_to_report(self):
        log_file = self.write_log("--- LLamaStudio Server Starting ---\n0.1 I srv all good\n")

        self.assertEqual(ServerManager._startup_errors(log_file), [])

    def test_missing_log_file_does_not_raise(self):
        self.assertEqual(ServerManager._startup_errors(Path("/nonexistent/server.log")), [])

    @patch("app.gpu_utils.get_gpu_info", return_value={"total_vram": 24.0})
    def test_memory_hint_describes_the_real_device(self, _mock_gpu):
        """The old hint hard-coded 32GB regardless of the card present."""
        self.assertIn("24 GiB", ServerManager._vram_description())

    @patch("app.gpu_utils.get_gpu_info", side_effect=OSError("no driver"))
    def test_memory_hint_survives_gpu_detection_failure(self, _mock_gpu):
        self.assertEqual(ServerManager._vram_description(), "available GPU memory")


class TestStaleErrorIsNotReported(unittest.TestCase):
    """A failed load must never be described by the previous load's error."""

    STALE = "invalid ggml type 142. should be in [0, 43)"

    def setUp(self):
        self.server = ServerManager()
        self.server._last_error = self.STALE
        self.addCleanup(setattr, self.server, "_last_error", None)

        # Keep these tests off the real log directory and the real process.
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        log_patch = patch.object(
            ServerManager, "_write_log", return_value=Path(self.directory) / "server.log"
        )
        log_patch.start()
        self.addCleanup(log_patch.stop)

    def test_a_missing_model_replaces_the_previous_error(self):
        with patch("app.model_manager.find_mmproj", return_value=""):
            loaded = self.server.load_model(f"{self.directory}/absent.gguf", {})

        self.assertFalse(loaded)
        self.assertNotEqual(self.server.last_error, self.STALE)
        self.assertIn("Model not found", self.server.last_error)

    def test_a_missing_projector_replaces_the_previous_error(self):
        model = Path(self.directory) / "present.gguf"
        model.write_bytes(b"GGUF")

        loaded = self.server.load_model(str(model), {"mmproj": f"{self.directory}/absent.mmproj"})

        self.assertFalse(loaded)
        self.assertNotEqual(self.server.last_error, self.STALE)
        self.assertIn("projector not found", self.server.last_error.lower())

    def test_an_exception_during_load_is_recorded(self):
        model = Path(self.directory) / "present.gguf"
        model.write_bytes(b"GGUF")

        with (
            patch("app.model_manager.find_mmproj", return_value=""),
            patch.object(ServerManager, "_build_command", side_effect=RuntimeError("boom")),
            patch.object(ServerManager, "eject_model", return_value=True),
        ):
            loaded = self.server.load_model(str(model), {})

        self.assertFalse(loaded)
        self.assertNotEqual(self.server.last_error, self.STALE)
        self.assertIn("boom", self.server.last_error)


if __name__ == "__main__":
    unittest.main()
