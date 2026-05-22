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
    LLAMA_SERVER_CTX_SIZE: int = 262144
    LLAMA_SERVER_GPU_LAYERS: int = 999
    LLAMA_SERVER_FLASH_ATTN: str = "on"
    LLAMA_SERVER_KV_CACHE_TYPE: str = "q8_0"
    LLAMA_SERVER_VOCAB_TYPE: str = "q8_0"
    LLAMA_SERVER_OVERRIDE_KV: str = "qwen35.context_length=int:262144"

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
    LOG_DIR: str = "/home/gnulnx/LLamaStuiod/logs"

    # Persistence files
    CONVERSATIONS_FILE: str = "/home/gnulnx/LLamaStuiod/app/conversations.json"
    MODEL_SETTINGS_FILE: str = "/home/gnulnx/LLamaStuiod/app/model_settings.json"

    model_config = {"env_prefix": "LLAMASTUDIO_"}

settings = Settings()

