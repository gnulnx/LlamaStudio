import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
