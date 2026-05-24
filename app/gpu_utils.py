"""
Utility for cross-platform GPU detection.
Works on Linux (Nvidia/AMD) and macOS (Apple Silicon).
"""

import platform
import re
import subprocess
from typing import TypedDict


class GPUInfo(TypedDict):
    name: str
    vram: int  # Total VRAM in GB


def get_gpu_info() -> GPUInfo:
    """
    Detects the primary GPU and its total VRAM.
    Returns a dictionary with name and vram (in GB).
    """
    sys_platform = platform.system()

    if sys_platform == "Darwin":
        # --- macOS (Unified Memory) ---
        try:
            output = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], text=True)
            name_match = re.search(r"Chip: (.+)", output)
            gpu_name = name_match.group(1).strip() if name_match else "Apple GPU"

            mem_output = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            vram_bytes = int(mem_output.strip())
            vram_gb = round(vram_bytes / (1024**3))

            return {"name": gpu_name, "vram": vram_gb}
        except Exception:
            return {"name": "Apple GPU", "vram": 8}

    elif sys_platform == "Linux":
        # --- Linux (Nvidia) ---
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                text=True,
            ).strip()
            if output:
                first_gpu = output.split("\n")[0]
                name, mem = first_gpu.split(",")
                return {"name": name.strip(), "vram": int(mem.strip()) // 1024}
        except Exception:
            pass

        # --- Linux (AMD/Radeon) ---
        try:
            output = subprocess.check_output(["rocm-smi", "--showmeminfo", "vram"], text=True)
            match = re.search(r"VRAM Total:\s+(\d+)", output)
            if match:
                vram_mb = int(match.group(1))
                try:
                    name_output = subprocess.check_output(
                        ["rocm-smi", "--showproductname"], text=True
                    )
                    gpu_name = name_output.strip() or "AMD Radeon GPU"
                except Exception:
                    gpu_name = "AMD Radeon GPU"
                return {"name": gpu_name, "vram": vram_mb // 1024}
        except Exception:
            pass

        # --- Linux (AMD/Radeon via sysfs /sys/class/drm) ---
        try:
            import glob

            vram_files = glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")
            if not vram_files:
                vram_files = glob.glob("/sys/class/drm/renderD*/device/mem_info_vram_total")
            if vram_files:
                with open(vram_files[0]) as f:
                    vram_bytes = int(f.read().strip())
                    vram_gb = round(vram_bytes / (1024**3))
                    if vram_gb > 0:
                        gpu_name = "AMD Radeon GPU"
                        try:
                            output = subprocess.check_output(["lspci"], text=True)
                            match = re.search(r"VGA compatible controller: (.+)", output)
                            if match:
                                gpu_name = match.group(1).strip()
                        except Exception:
                            pass
                        return {"name": gpu_name, "vram": vram_gb}
        except Exception:
            pass

        # --- Linux Generic Fallback (lspci) ---
        try:
            output = subprocess.check_output(["lspci"], text=True)
            match = re.search(r"VGA compatible controller: (.+)", output)
            if match:
                return {"name": match.group(1).strip(), "vram": 8}
        except Exception:
            pass

    return {"name": "Generic GPU", "vram": 8}
