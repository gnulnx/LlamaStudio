"""
Utility for cross-platform GPU detection.
Works on Linux (Nvidia/AMD) and macOS (Apple Silicon).
"""

from __future__ import annotations

import platform
import re
import subprocess
from math import isfinite
from typing import TypedDict


class GPUIdentity(TypedDict):
    name: str
    vram: int  # Total VRAM in GB


class GPUInfo(GPUIdentity, total=False):
    total_vram: float | None  # GiB; None when detection is only a guess
    used_vram: float | None  # Device-wide usage, not just llama-server
    free_vram: float | None
    memory_kind: str  # "dedicated" or "unified"


def _memory_info(total: float | None, used: float | None = None) -> dict:
    """Never turn a missing/invalid measurement into an apparently idle GPU."""
    if total is not None and (not isfinite(total) or total <= 0):
        total = None
    if total is None or used is None or not 0 <= used <= total:
        used = None
    return {
        "total_vram": total,
        "used_vram": used,
        "free_vram": total - used if total is not None and used is not None else None,
    }


def _query(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, timeout=2, stderr=subprocess.DEVNULL).strip()


def get_gpu_info() -> GPUInfo:
    """
    Detect the primary GPU, capacity, and available live memory measurements.
    Keep the legacy name/vram fields for the web UI and older clients. Optional
    measurements use GiB and remain unknown when a driver cannot report them.
    """
    sys_platform = platform.system()

    if sys_platform == "Darwin":
        # --- macOS (Unified Memory) ---
        try:
            output = _query(["system_profiler", "SPDisplaysDataType"])
            name_match = re.search(r"Chip: (.+)", output)
            gpu_name = name_match.group(1).strip() if name_match else "Apple GPU"

            mem_output = _query(["sysctl", "-n", "hw.memsize"])
            vram_bytes = int(mem_output.strip())
            vram_gb = round(vram_bytes / (1024**3))

            return {
                "name": gpu_name,
                "vram": vram_gb,
                "memory_kind": "unified",
                **_memory_info(vram_bytes / (1024**3)),
            }
        except Exception:
            return {"name": "Apple GPU", "vram": 8, "memory_kind": "unified", **_memory_info(None)}

    elif sys_platform == "Linux":
        # --- Linux (Nvidia) ---
        try:
            output = _query([
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ])
            if output:
                first_gpu = output.split("\n")[0]
                name, mem, used = first_gpu.split(",")
                total_gib = float(mem.strip()) / 1024
                try:
                    used_gib = float(used.strip()) / 1024
                except ValueError:
                    used_gib = None
                return {
                    "name": name.strip(),
                    "vram": int(total_gib),
                    "memory_kind": "dedicated",
                    **_memory_info(total_gib, used_gib),
                }
        except Exception:
            pass

        # --- Linux (AMD/Radeon) ---
        try:
            output = _query(["rocm-smi", "--showmeminfo", "vram"])
            match = re.search(r"VRAM Total:\s+(\d+)", output)
            if match:
                vram_mb = int(match.group(1))
                try:
                    name_output = _query(["rocm-smi", "--showproductname"])
                    gpu_name = name_output.strip() or "AMD Radeon GPU"
                except Exception:
                    gpu_name = "AMD Radeon GPU"
                return {
                    "name": gpu_name,
                    "vram": vram_mb // 1024,
                    **_memory_info(vram_mb / 1024),
                }
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
                            output = _query(["lspci"])
                            match = re.search(r"VGA compatible controller: (.+)", output)
                            if match:
                                gpu_name = match.group(1).strip()
                        except Exception:
                            pass
                        return {
                            "name": gpu_name,
                            "vram": vram_gb,
                            **_memory_info(vram_bytes / (1024**3)),
                        }
        except Exception:
            pass

        # --- Linux Generic Fallback (lspci) ---
        try:
            output = _query(["lspci"])
            match = re.search(r"VGA compatible controller: (.+)", output)
            if match:
                return {"name": match.group(1).strip(), "vram": 8, **_memory_info(None)}
        except Exception:
            pass

    return {"name": "Generic GPU", "vram": 8, **_memory_info(None)}
