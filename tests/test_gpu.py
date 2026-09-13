import asyncio
import os
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

# Ensure the app package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.gpu_utils import get_gpu_info


class TestGPUUtils(unittest.TestCase):
    """Test suite for app/gpu_utils.py GPU detection helper."""

    @patch("app.gpu_utils.platform.system", return_value="Linux")
    @patch("app.gpu_utils.subprocess.check_output")
    def test_linux_nvidia_success(self, mock_subprocess, mock_system):
        # Mock successful nvidia-smi command execution
        mock_subprocess.return_value = "NVIDIA GeForce RTX 5090, 32607, 25385\n"

        info = get_gpu_info()
        self.assertEqual(info["name"], "NVIDIA GeForce RTX 5090")
        self.assertEqual(info["vram"], 31)
        self.assertEqual(info["total_vram"], 32607 / 1024)
        self.assertEqual(info["used_vram"], 25385 / 1024)
        self.assertEqual(info["free_vram"], (32607 - 25385) / 1024)
        mock_subprocess.assert_called_once_with(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
            stderr=subprocess.DEVNULL,
        )

    @patch("app.gpu_utils.platform.system", return_value="Linux")
    @patch("app.gpu_utils.subprocess.check_output")
    def test_nvidia_unknown_or_invalid_usage_is_not_zero(self, mock_subprocess, mock_system):
        for used in ("[N/A]", "-1", "65536", "nan", "inf"):
            with self.subTest(used=used):
                mock_subprocess.return_value = f"NVIDIA RTX, 32768, {used}\n"
                info = get_gpu_info()
                self.assertEqual(info["total_vram"], 32)
                self.assertIsNone(info["used_vram"])
                self.assertIsNone(info["free_vram"])

    @patch("app.gpu_utils.platform.system", return_value="Linux")
    @patch("app.gpu_utils.subprocess.check_output")
    def test_nvidia_reports_primary_device_not_sum_of_devices(self, mock_subprocess, mock_system):
        mock_subprocess.return_value = "GPU 0, 24576, 1024\nGPU 1, 32768, 16384\n"
        info = get_gpu_info()
        self.assertEqual(info["name"], "GPU 0")
        self.assertEqual(info["total_vram"], 24)
        self.assertEqual(info["used_vram"], 1)

    @patch("app.gpu_utils.platform.system", return_value="Linux")
    @patch("glob.glob", return_value=[])
    @patch(
        "app.gpu_utils.subprocess.check_output", side_effect=subprocess.TimeoutExpired("probe", 2)
    )
    def test_unavailable_detection_does_not_report_guessed_capacity(self, *mocks):
        info = get_gpu_info()
        self.assertIsNone(info["total_vram"])
        self.assertIsNone(info["used_vram"])

    @patch("app.gpu_utils.platform.system", return_value="Linux")
    @patch("app.gpu_utils.subprocess.check_output")
    def test_linux_nvidia_fail_amd_success(self, mock_subprocess, mock_system):
        # First call (nvidia-smi) fails, second call (rocm-smi) succeeds
        def check_output_side_effect(args, **kwargs):
            if "nvidia-smi" in args[0]:
                raise FileNotFoundError()
            elif "rocm-smi" in args[0]:
                if "--showmeminfo" in args:
                    return "VRAM Total: 16384\n"
                elif "--showproductname" in args:
                    return "Radeon RX 7900 XTX\n"
            raise FileNotFoundError()

        mock_subprocess.side_effect = check_output_side_effect

        info = get_gpu_info()
        self.assertEqual(info["name"], "Radeon RX 7900 XTX")
        self.assertEqual(info["vram"], 16)

    @patch("app.gpu_utils.platform.system", return_value="Linux")
    @patch("app.gpu_utils.subprocess.check_output")
    @patch("glob.glob")
    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data="6442450944\n")
    def test_linux_amd_sysfs_success(self, mock_file, mock_glob, mock_subprocess, mock_system):
        # nvidia-smi and rocm-smi fail, glob returns sysfs files, lspci succeeds
        def check_output_side_effect(args, **kwargs):
            if "lspci" in args[0]:
                return "65:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Phoenix1 (rev c2)\n"
            raise FileNotFoundError()

        mock_subprocess.side_effect = check_output_side_effect
        mock_glob.return_value = ["/sys/class/drm/card0/device/mem_info_vram_total"]

        info = get_gpu_info()
        self.assertEqual(info["name"], "Advanced Micro Devices, Inc. [AMD/ATI] Phoenix1 (rev c2)")
        self.assertEqual(info["vram"], 6)

    @patch("app.gpu_utils.platform.system", return_value="Linux")
    @patch("app.gpu_utils.subprocess.check_output")
    @patch("glob.glob", return_value=[])
    def test_linux_fallback_lspci(self, mock_glob, mock_subprocess, mock_system):
        # Both nvidia-smi and rocm-smi fail, lspci succeeds
        def check_output_side_effect(args, **kwargs):
            if "lspci" in args[0]:
                return "01:00.0 VGA compatible controller: NVIDIA Corporation AD102 [GeForce RTX 4090] (rev a1)\n"
            raise FileNotFoundError()

        mock_subprocess.side_effect = check_output_side_effect

        info = get_gpu_info()
        self.assertEqual(info["name"], "NVIDIA Corporation AD102 [GeForce RTX 4090] (rev a1)")
        self.assertEqual(info["vram"], 8)

    @patch("app.gpu_utils.platform.system", return_value="Darwin")
    @patch("app.gpu_utils.subprocess.check_output")
    def test_macos_apple_silicon(self, mock_subprocess, mock_system):
        def check_output_side_effect(args, **kwargs):
            if "system_profiler" in args[0]:
                return "Chip: Apple M3 Max\n"
            elif "sysctl" in args[0]:
                # 32 GB RAM in bytes
                return "34359738368\n"
            raise FileNotFoundError()

        mock_subprocess.side_effect = check_output_side_effect

        info = get_gpu_info()
        self.assertEqual(info["name"], "Apple M3 Max")
        self.assertEqual(info["vram"], 32)
        self.assertEqual(info["total_vram"], 32)
        self.assertEqual(info["memory_kind"], "unified")
        self.assertIsNone(info["used_vram"])
        self.assertIsNone(info["free_vram"])


class TestGPUEndpoint(unittest.TestCase):
    """Test suite for the GET /api/gpu endpoint."""

    @patch("app.gpu_utils.get_gpu_info")
    def test_gpu_endpoint(self, mock_get_gpu_info):
        from app.main import get_gpu

        expected_response = {"name": "NVIDIA GeForce RTX 5090", "vram": 31}
        event_loop_thread = threading.get_ident()

        def probe():
            self.assertNotEqual(threading.get_ident(), event_loop_thread)
            return expected_response

        mock_get_gpu_info.side_effect = probe

        response = asyncio.run(get_gpu())
        self.assertEqual(response, expected_response)
        mock_get_gpu_info.assert_called_once()


if __name__ == "__main__":
    unittest.main()
