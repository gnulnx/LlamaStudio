"""
Application configuration.
Edit these values to match your setup.
"""

import shutil
from pathlib import Path

from pydantic_settings import BaseSettings


def resolve_llama_server_bin(force_cpu: bool = False) -> str:
    """Dynamically search for the best llama-server binary based on hardware and force_cpu setting."""
    home = Path.home()
    bin_root = home / "llama.cpp-bin"

    if force_cpu:
        cpu_path = bin_root / "cpu" / "llama-server"
        if cpu_path.exists():
            return str(cpu_path)

    # Probing GPU vendor
    vendor = "unknown"
    for card_dir in ["card0", "card1", "card2"]:
        vendor_path = Path(f"/sys/class/drm/{card_dir}/device/vendor")
        if vendor_path.exists():
            try:
                vendor_id = vendor_path.read_text().strip().lower()
                if "1002" in vendor_id:
                    vendor = "amd"
                    break
                elif "10de" in vendor_id:
                    vendor = "nvidia"
                    break
            except Exception:
                pass

    if vendor == "unknown":
        try:
            import subprocess

            lspci = subprocess.check_output(["lspci"], stderr=subprocess.DEVNULL, text=True).lower()
            if "nvidia" in lspci:
                vendor = "nvidia"
            elif "amd" in lspci or "ati" in lspci:
                vendor = "amd"
        except Exception:
            pass

    # Resolve based on vendor
    if vendor == "nvidia" and not force_cpu:
        cuda_path = bin_root / "cuda" / "llama-server"
        if cuda_path.exists():
            return str(cuda_path)
        # Fallback to Vulkan if CUDA not found
        vulkan_path = bin_root / "vulkan" / "llama-server"
        if vulkan_path.exists():
            return str(vulkan_path)

    elif vendor == "amd" and not force_cpu:
        # Check ROCm first, then Vulkan
        rocm_path = bin_root / "rocm" / "llama-server"
        if rocm_path.exists():
            return str(rocm_path)
        vulkan_path = bin_root / "vulkan" / "llama-server"
        if vulkan_path.exists():
            return str(vulkan_path)

    # Standard fallback path scans
    if not force_cpu:
        for folder in ["cuda", "rocm", "vulkan", "cpu"]:
            p = bin_root / folder / "llama-server"
            if p.exists():
                return str(p)

    # Check global PATH or custom build directories
    for binary in ["llama-server", "llama.cpp-server"]:
        resolved = shutil.which(binary)
        if resolved:
            return resolved

    possible_paths = [
        home / "llama.cpp" / "build" / "bin" / "llama-server",
        home / "llama.cpp" / "llama-server",
    ]
    for p in possible_paths:
        if p.exists():
            return str(p)

    fallback_cpu = bin_root / "cpu" / "llama-server"
    if fallback_cpu.exists():
        return str(fallback_cpu)

    return str(home / "llama.cpp" / "build" / "bin" / "llama-server")


class Settings(BaseSettings):
    # llama.cpp server settings
    LLAMA_SERVER_BIN: str = resolve_llama_server_bin()
    LLAMA_SERVER_PORT: int = 1234
    LLAMA_SERVER_CTX_SIZE: int = 16384
    LLAMA_SERVER_GPU_LAYERS: int = 999
    LLAMA_SERVER_FLASH_ATTN: str = "on"
    LLAMA_SERVER_KV_CACHE_TYPE: str = "q8_0"
    LLAMA_SERVER_VOCAB_TYPE: str = "q8_0"
    LLAMA_SERVER_OVERRIDE_KV: str = ""
    LLAMA_SERVER_TASK_TIMEOUT: int = 900

    # Default fallback model path (unused, starts clean without loaded model)
    DEFAULT_MODEL: str = ""

    # Model directories to scan (portable user home folder)
    MODEL_DIRS: list[str] = [
        str(Path.home() / ".lmstudio" / "models"),
    ]

    # Chat settings
    DEFAULT_SYSTEM_PROMPT: str = "You are a helpful assistant."
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_TOP_P: float = 0.9
    DEFAULT_MAX_TOKENS: int = 2048
    CHAT_REQUEST_TIMEOUT: int = 900
    MAX_TOOL_ITERATIONS: int = 50

    # App settings
    APP_PORT: int = 8765
    APP_HOST: str = "127.0.0.1"

    # Sandbox/Workspace root for tools
    WORKSPACE_ROOT: str = str(Path(__file__).parent.parent.resolve())
    DISABLE_SANDBOX: bool = False

    # Runtime files stored outside the repository/package in the user's config home
    CONFIG_DIR: str = str(Path.home() / ".config" / "llamastudio")
    CONFIG_FILE: str = str(Path.home() / ".config" / "llamastudio" / "config.json")
    LOG_DIR: str = str(Path.home() / ".config" / "llamastudio" / "logs")
    CONVERSATIONS_FILE: str = str(Path.home() / ".config" / "llamastudio" / "conversations.json")
    MODEL_PROFILES_FILE: str = str(
        Path.home() / ".config" / "llamastudio" / "model_profiles.json"
    )
    MODEL_SETTINGS_FILE: str = str(Path.home() / ".config" / "llamastudio" / "model_settings.json")

    model_config = {
        "env_prefix": "LLAMASTUDIO_",
    }

    @classmethod
    def migrate_persistence_files(cls):
        """Migrate any old persistence files from the app directory to ~/.config/llamastudio/"""
        import shutil

        old_dir = Path(__file__).parent
        new_dir = Path.home() / ".config" / "llamastudio"

        for filename in ["conversations.json", "model_settings.json"]:
            old_path = old_dir / filename
            new_path = new_dir / filename

            if old_path.exists() and not new_path.exists():
                try:
                    new_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(old_path, new_path)
                except Exception:
                    pass


# Run config migrations
Settings.migrate_persistence_files()

settings = Settings()
