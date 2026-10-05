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


class SingleDeviceInfo(TypedDict, total=False):
    id: str
    index: int
    name: str
    vram: int  # Total VRAM in GB
    total_vram: float | None  # GiB
    used_vram: float | None
    free_vram: float | None
    memory_kind: str  # "dedicated" or "unified"


class GPUInfo(GPUIdentity, total=False):
    total_vram: float | None  # GiB; None when detection is only a guess
    used_vram: float | None  # Device-wide usage, not just llama-server
    free_vram: float | None
    memory_kind: str  # "dedicated" or "unified"
    devices: list[SingleDeviceInfo]
    device_count: int
    total_system_vram: float | None
    total_system_free: float | None


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
    Detect all available GPUs, capacities, and live memory measurements.
    Keep legacy primary name/vram fields for backward compatibility while exposing
    a structured 'devices' list and aggregated 'total_system_vram'.
    """
    sys_platform = platform.system()

    if sys_platform == "Darwin":
        # --- macOS (Unified Memory) ---
        try:
            output = _query(["system_profiler", "SPDisplaysDataType"])
            name_matches = re.findall(r"Chip: (.+)", output)
            gpu_name = name_matches[0].strip() if name_matches else "Apple GPU"

            mem_output = _query(["sysctl", "-n", "hw.memsize"])
            vram_bytes = int(mem_output.strip())
            vram_gb = round(vram_bytes / (1024**3))
            primary_mem = _memory_info(vram_bytes / (1024**3))

            devices: list[SingleDeviceInfo] = []
            if name_matches:
                for idx, chip in enumerate(name_matches):
                    devices.append({
                        "id": str(idx),
                        "index": idx,
                        "name": chip.strip(),
                        "vram": vram_gb,
                        "memory_kind": "unified",
                        **primary_mem,
                    })
            else:
                devices.append({
                    "id": "0",
                    "index": 0,
                    "name": gpu_name,
                    "vram": vram_gb,
                    "memory_kind": "unified",
                    **primary_mem,
                })

            return {
                "name": gpu_name,
                "vram": vram_gb,
                "memory_kind": "unified",
                "devices": devices,
                "device_count": len(devices),
                "total_system_vram": float(vram_gb),
                "total_system_free": None,
                **primary_mem,
            }
        except Exception:
            fallback = {
                "name": "Apple GPU",
                "vram": 8,
                "memory_kind": "unified",
                **_memory_info(None),
            }
            return {
                **fallback,
                "devices": [{"id": "0", "index": 0, **fallback}],
                "device_count": 1,
                "total_system_vram": 8.0,
                "total_system_free": None,
            }

    elif sys_platform == "Linux":
        # --- Linux (Nvidia) ---
        try:
            output = _query([
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ])
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            if lines:
                devices: list[SingleDeviceInfo] = []
                for idx, line in enumerate(lines):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        name, mem, used = parts[0], parts[1], parts[2]
                        try:
                            total_gib = float(mem) / 1024
                        except ValueError:
                            total_gib = None
                        try:
                            used_gib = float(used) / 1024
                        except ValueError:
                            used_gib = None
                        mem_info = _memory_info(total_gib, used_gib)
                        devices.append({
                            "id": str(idx),
                            "index": idx,
                            "name": name,
                            "vram": int(total_gib) if total_gib is not None else 8,
                            "memory_kind": "dedicated",
                            **mem_info,
                        })
                if devices:
                    primary = devices[0]
                    valid_totals = [
                        d["total_vram"] for d in devices if d.get("total_vram") is not None
                    ]
                    valid_frees = [
                        d["free_vram"] for d in devices if d.get("free_vram") is not None
                    ]
                    total_sys = sum(valid_totals) if valid_totals else None
                    total_free = (
                        sum(valid_frees)
                        if len(valid_frees) == len(devices) and valid_frees
                        else None
                    )
                    return {
                        "name": primary["name"],
                        "vram": primary["vram"],
                        "memory_kind": primary["memory_kind"],
                        "total_vram": primary["total_vram"],
                        "used_vram": primary["used_vram"],
                        "free_vram": primary["free_vram"],
                        "devices": devices,
                        "device_count": len(devices),
                        "total_system_vram": total_sys,
                        "total_system_free": total_free,
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
                mem_info = _memory_info(vram_mb / 1024)
                device: SingleDeviceInfo = {
                    "id": "0",
                    "index": 0,
                    "name": gpu_name,
                    "vram": vram_mb // 1024,
                    "memory_kind": "dedicated",
                    **mem_info,
                }
                return {
                    "name": gpu_name,
                    "vram": vram_mb // 1024,
                    "memory_kind": "dedicated",
                    "devices": [device],
                    "device_count": 1,
                    "total_system_vram": float(vram_mb / 1024),
                    "total_system_free": mem_info.get("free_vram"),
                    **mem_info,
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
                devices: list[SingleDeviceInfo] = []
                for idx, vram_file in enumerate(vram_files):
                    with open(vram_file) as f:
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
                            mem_info = _memory_info(vram_bytes / (1024**3))
                            devices.append({
                                "id": str(idx),
                                "index": idx,
                                "name": gpu_name,
                                "vram": vram_gb,
                                "memory_kind": "dedicated",
                                **mem_info,
                            })
                if devices:
                    primary = devices[0]
                    valid_totals = [
                        d["total_vram"] for d in devices if d.get("total_vram") is not None
                    ]
                    valid_frees = [
                        d["free_vram"] for d in devices if d.get("free_vram") is not None
                    ]
                    total_sys = sum(valid_totals) if valid_totals else None
                    total_free = (
                        sum(valid_frees)
                        if len(valid_frees) == len(devices) and valid_frees
                        else None
                    )
                    return {
                        "name": primary["name"],
                        "vram": primary["vram"],
                        "memory_kind": primary["memory_kind"],
                        "total_vram": primary["total_vram"],
                        "used_vram": primary["used_vram"],
                        "free_vram": primary["free_vram"],
                        "devices": devices,
                        "device_count": len(devices),
                        "total_system_vram": total_sys,
                        "total_system_free": total_free,
                    }
        except Exception:
            pass

        # --- Linux Generic Fallback (lspci) ---
        try:
            output = _query(["lspci"])
            match = re.search(r"VGA compatible controller: (.+)", output)
            if match:
                name = match.group(1).strip()
                device = {
                    "id": "0",
                    "index": 0,
                    "name": name,
                    "vram": 8,
                    "memory_kind": "dedicated",
                    **_memory_info(None),
                }
                return {
                    "name": name,
                    "vram": 8,
                    "memory_kind": "dedicated",
                    "devices": [device],
                    "device_count": 1,
                    "total_system_vram": 8.0,
                    "total_system_free": None,
                    **_memory_info(None),
                }
        except Exception:
            pass

    fallback_device: SingleDeviceInfo = {
        "id": "0",
        "index": 0,
        "name": "Generic GPU",
        "vram": 8,
        "memory_kind": "dedicated",
        **_memory_info(None),
    }
    return {
        "name": "Generic GPU",
        "vram": 8,
        "memory_kind": "dedicated",
        "devices": [fallback_device],
        "device_count": 1,
        "total_system_vram": 8.0,
        "total_system_free": None,
        **_memory_info(None),
    }
