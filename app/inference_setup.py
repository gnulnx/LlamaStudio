"""Checksum-pinned, private llama.cpp installations for Linux and macOS."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from .inference_runtime import inference_root, managed_backend, runtime_environment

RELEASE = "b11565"
RELEASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}"
# Digests and lengths from the upstream GitHub release API. Never use /latest:
# upstream also publishes placeholder releases without platform binaries.
ASSETS = {
    "ubuntu-x64": (17712716, "c571742335d8b3d9a7a15773907d1b0c7d948ece274e88fa17bd77176ad52638"),
    "ubuntu-arm64": (13828645, "6a862c22bd58f8f86168e0ef7ed875166ad625d42f7cac9825d63b8c55af5c1e"),
    "ubuntu-vulkan-x64": (
        30274774,
        "da7b5c0910ffe48436ec6d934b74700eb213b94d6d5b4361e596e5e71c6185fa",
    ),
    "ubuntu-vulkan-arm64": (
        24051707,
        "bc9e30fc7bd13ee1c771b531c5aeb0e1742c03c9333260e138cb820f69d8d16e",
    ),
    "ubuntu-cuda-12.8-x64": (
        171559844,
        "61b8cf4e49739afe4008b06a92390dff999e706e9e7e9749c015544a9f241627",
    ),
    "cudart-ubuntu-cuda-12.8-x64": (
        594377834,
        "e6177074761d8be20b76ce165944f16b7a338c0b477d1ecbb240e38c743c1b90",
    ),
    "macos-arm64": (12105509, "19289548468babcd5131cecccf3a9bc7c67030cc7f0a978dafc0db1ba7891068"),
    "macos-x64": (11596850, "853c2c2ed9d796f3f6d0210aac7768a8398997cf2afdc9916a6112003b13760f"),
}
BACKENDS = ("auto", "cuda", "vulkan", "metal", "cpu")
_INSTALL_LOCK = threading.Lock()


class InferenceError(RuntimeError):
    """An actionable local inference setup failure."""


@dataclass
class Diagnosis:
    binary: str
    usable: bool
    version: str = ""
    devices: str = ""
    problem: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def diagnose(binary: str | None = None, *, backend: str = "auto") -> Diagnosis:
    if binary is None:
        from .config import resolve_llama_server_bin

        binary = resolve_llama_server_bin()
    if backend == "auto":
        backend = managed_backend(binary) or recommended_backend()
    result = Diagnosis(binary=binary, usable=False)
    if not Path(binary).is_file():
        result.problem = "llama-server is missing. Run lls setup."
        return result
    env = runtime_environment(binary)
    for flag, field in (("--version", "version"), ("--list-devices", "devices")):
        try:
            completed = subprocess.run(
                [binary, flag], env=env, text=True, capture_output=True, timeout=30
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result.problem = f"Cannot run llama-server: {exc}. Run lls setup --repair."
            return result
        output = (completed.stdout + completed.stderr).strip()
        if completed.returncode:
            result.problem = (
                f"llama-server {flag} failed: {output[-2000:]}. "
                "Run lls setup --repair; GPU backends also require a working system driver."
            )
            return result
        setattr(result, field, output)
    # Actual device rows, not build/startup messages announcing a backend.
    if backend in ("cuda", "vulkan", "metal") and not any(
        line.strip().lower().startswith(backend) for line in result.devices.splitlines()
    ):
        result.problem = (
            f"No {backend.upper()} device is available. Check the system GPU driver "
            "or explicitly choose lls setup --backend cpu."
        )
        return result
    result.usable = True
    return result


def recommended_backend() -> str:
    if platform.system() == "Darwin":
        return "metal" if platform.machine().lower() in ("arm64", "aarch64") else "cpu"
    try:
        result = subprocess.run(["lspci"], text=True, capture_output=True, timeout=5, check=False)
        hardware = "\n".join(
            line.lower()
            for line in result.stdout.splitlines()
            if any(kind in line.lower() for kind in ("vga", "3d controller", "display controller"))
        )
    except (OSError, subprocess.TimeoutExpired):
        hardware = ""
    if "nvidia" in hardware:
        return "cuda"
    if "amd" in hardware or ("intel" in hardware and ("vga" in hardware or "display" in hardware)):
        return "vulkan"
    return "cpu"


def selected_assets(backend: str) -> tuple[str, list[str]]:
    if backend not in BACKENDS:
        raise InferenceError(f"Unknown backend: {backend}")
    backend = recommended_backend() if backend == "auto" else backend
    system = platform.system()
    machine = platform.machine().lower()
    arch = {"x86_64": "x64", "amd64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine)
    if system == "Darwin" and arch and backend in ("metal", "cpu"):
        return backend, [f"macos-{arch}"]
    if system == "Linux" and arch:
        if backend == "cpu":
            return backend, [f"ubuntu-{arch}"]
        if backend == "vulkan":
            return backend, [f"ubuntu-vulkan-{arch}"]
        if backend == "cuda" and arch == "x64":
            return backend, ["ubuntu-cuda-12.8-x64", "cudart-ubuntu-cuda-12.8-x64"]
    raise InferenceError(
        f"No managed {backend} bundle for {system}/{machine}. "
        "Use --backend cpu on supported Linux/macOS hosts, or install llama.cpp manually "
        "and set LLAMASTUDIO_LLAMA_SERVER_BIN."
    )


@contextmanager
def _safe_scope(root: Path):
    # Lazy import avoids config -> runtime discovery -> config_store cycles.
    from .tools import reset_current_workspace_root, set_current_workspace_root

    token = set_current_workspace_root(root)
    try:
        yield
    finally:
        reset_current_workspace_root(token)


def _safe(path: Path) -> Path:
    from .tools import check_path_safe, get_effective_workspace_root

    resolved = check_path_safe(str(path))
    if not resolved.is_relative_to(get_effective_workspace_root()):
        raise InferenceError("Unsafe inference path")
    return resolved


def _download(asset: str, target: Path, progress: Callable[[str], None]) -> None:
    expected_size, checksum = ASSETS[asset]
    prefix = "cudart-" if asset.startswith("cudart-") else ""
    suffix = asset.removeprefix("cudart-")
    filename = f"{prefix}llama-{RELEASE}-bin-{suffix}.tar.gz"
    progress(f"Downloading {filename} ({expected_size / 1024**2:.0f} MiB)")
    digest = hashlib.sha256()
    count = 0
    with httpx.stream(
        "GET", f"{RELEASE_URL}/{filename}", follow_redirects=True, timeout=120
    ) as response:
        response.raise_for_status()
        with _safe(target).open("wb") as output:
            for chunk in response.iter_bytes(1024 * 1024):
                count += len(chunk)
                if count > expected_size:
                    raise InferenceError(f"Download size mismatch: {filename}")
                digest.update(chunk)
                output.write(chunk)
    if count != expected_size or digest.hexdigest() != checksum:
        raise InferenceError(f"Download size or SHA256 mismatch: {filename}")


def _extract(archive: Path, destination: Path) -> None:
    """Extract only confined regular files, directories and relative symlinks."""
    with _safe_scope(destination), tarfile.open(_safe(archive), "r:gz") as bundle:
        total = 0
        for member in bundle:
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise InferenceError("Unsafe archive path")
            target = _safe(destination / name)
            total += member.size
            if total > 4 * 1024**3:
                raise InferenceError("Archive exceeds extraction limit")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = bundle.extractfile(member)
                if stream is None:
                    raise InferenceError("Missing archive file")
                with stream, target.open("wb") as output:
                    shutil.copyfileobj(stream, output)
                target.chmod(member.mode & 0o755)
            elif member.issym():
                link = Path(member.linkname)
                if link.is_absolute():
                    raise InferenceError("Unsafe archive symlink")
                _safe(target.parent / link)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(link)
            else:
                raise InferenceError("Unsupported archive entry")


def setup(
    *, backend: str = "auto", repair: bool = False, progress: Callable[[str], None] | None = None
) -> Diagnosis:
    """Validate before activation; never modify external builds or system drivers."""
    progress = progress or (lambda message: None)
    current = diagnose()
    if current.usable and not repair:
        progress(f"Keeping working llama-server: {current.binary}")
        return current
    if os.environ.get("LLAMASTUDIO_LLAMA_SERVER_BIN"):
        raise InferenceError(
            "LLAMASTUDIO_LLAMA_SERVER_BIN explicitly selects an external server. "
            "Repair that installation or unset the override before managed setup."
        )
    backend, assets = selected_assets(backend)
    if not _INSTALL_LOCK.acquire(blocking=False):
        raise InferenceError("Inference setup is already running.")
    root = inference_root().expanduser()
    stage = None
    try:
        root = root.resolve()
        with _safe_scope(root):
            releases = _safe(root / "releases")
            releases.mkdir(parents=True, exist_ok=True)
            # Cross-process lock as well as the in-process lock; flock works on macOS.
            import fcntl

            with _safe(root / ".setup.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise InferenceError("Inference setup is already running.") from exc
                stage = Path(tempfile.mkdtemp(prefix=f"{RELEASE}-{backend}-", dir=releases))
                for index, asset in enumerate(assets):
                    archive = _safe(stage / f"bundle-{index}.tar.gz")
                    _download(asset, archive, progress)
                    _extract(archive, stage)
                    archive.unlink()
                binaries = list(stage.rglob("llama-server"))
                if len(binaries) != 1:
                    raise InferenceError("Release does not contain exactly one llama-server")
                binary = _safe(binaries[0])
                progress("Checking llama-server and available devices")
                diagnosis = diagnose(str(binary), backend=backend)
                if not diagnosis.usable:
                    raise InferenceError(diagnosis.problem)
                manifest = _safe(root / "current.json")
                pending = _safe(root / "current.json.tmp")
                pending.write_text(
                    json.dumps({
                        "release": RELEASE,
                        "backend": backend,
                        "binary": str(binary.relative_to(root)),
                    })
                    + "\n"
                )
                pending.replace(manifest)
                stage = None  # Successful installs and previous versions are retained.
                return diagnosis
    except (OSError, ValueError, tarfile.TarError, httpx.HTTPError) as exc:
        raise InferenceError(f"Inference setup failed: {exc}") from exc
    finally:
        if stage is not None:
            with _safe_scope(root):
                shutil.rmtree(_safe(stage))
        _INSTALL_LOCK.release()
