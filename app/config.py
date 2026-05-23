"""
Application configuration.
Edit these values to match your setup.
"""
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # llama.cpp server settings
    LLAMA_SERVER_BIN: str = "/home/gnulnx/llama.cpp/build/bin/llama-server"
    LLAMA_SERVER_PORT: int = 1234
    LLAMA_SERVER_CTX_SIZE: int = 16384
    LLAMA_SERVER_GPU_LAYERS: int = 999
    LLAMA_SERVER_FLASH_ATTN: str = "on"
    LLAMA_SERVER_KV_CACHE_TYPE: str = "q8_0"
    LLAMA_SERVER_VOCAB_TYPE: str = "q8_0"
    LLAMA_SERVER_OVERRIDE_KV: str = ""

    # Default model path
    DEFAULT_MODEL: str = "/home/gnulnx/.lmstudio/models/Jackrong/Qwopus3.6-27B-v2-GGUF/Qwopus3.6-27B-v2-Q4_K_S.gguf"

    # Model directories to scan
    MODEL_DIRS: list[str] = [
        "/home/gnulnx/.lmstudio/models",
    ]

    # Chat settings
    DEFAULT_SYSTEM_PROMPT: str = "You are a helpful assistant."
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_TOP_P: float = 0.9
    DEFAULT_MAX_TOKENS: int = 2048

    # App settings
    APP_PORT: int = 8765
    APP_HOST: str = "127.0.0.1"

    # Log file for llama-server output
    LOG_DIR: str = str(Path(__file__).parent.parent / "logs")

    # Persistence files stored outside the repository in the user's config home
    CONVERSATIONS_FILE: str = str(Path.home() / ".config" / "llamastudio" / "conversations.json")
    MODEL_SETTINGS_FILE: str = str(Path.home() / ".config" / "llamastudio" / "model_settings.json")

    model_config = {"env_prefix": "LLAMASTUDIO_"}

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

