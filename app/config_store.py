"""
First-class runtime configuration and model profile persistence.

The pydantic settings object still provides bootstrap defaults and environment
overrides. ConfigLoader owns user-facing app state stored under the config dir.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import settings
from .logger import logger

SCHEMA_VERSION = 1


class ConfigLoader:
    """Load, migrate, and persist LlamaStudio config and model profiles."""

    def __init__(
        self,
        config_dir: str | Path | None = None,
        legacy_model_settings_file: str | Path | None = None,
    ):
        self.config_dir = Path(config_dir or settings.CONFIG_DIR).expanduser()
        self.config_file = self.config_dir / "config.json"
        if config_dir is None:
            self.model_profiles_file = Path(settings.MODEL_PROFILES_FILE).expanduser()
        else:
            self.model_profiles_file = self.config_dir / "model_profiles.json"
        self.legacy_model_settings_file = Path(
            legacy_model_settings_file or settings.MODEL_SETTINGS_FILE
        ).expanduser()

    def ensure_initialized(self, workspace_root: str | Path | None = None) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        (self.config_dir / "logs").mkdir(parents=True, exist_ok=True)

        if not self.config_file.exists():
            self._write_json(self.config_file, self._default_config(workspace_root))
        else:
            self.save_app_config(self._merge_app_config(self._read_json(self.config_file, {})))

        if not self.model_profiles_file.exists():
            profiles = self._migrate_legacy_model_settings()
            self._write_json(self.model_profiles_file, profiles)

    def initialize_for_launch(self, workspace_root: str | Path | None = None) -> None:
        self.ensure_initialized(workspace_root=workspace_root)

    def load_app_config(self) -> dict[str, Any]:
        self.ensure_initialized()
        data = self._read_json(self.config_file, {})
        return self._merge_app_config(data)

    def save_app_config(self, config: dict[str, Any]) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(self.config_file, self._merge_app_config(config))

    def get_workspace_config(self) -> dict[str, Any]:
        return deepcopy(self.load_app_config()["workspace"])

    def get_workspace_root(self) -> str:
        return str(Path(self.get_workspace_config()["root"]).expanduser())

    def sandbox_disabled(self) -> bool:
        return bool(self.get_workspace_config().get("disable_sandbox", False))

    def get_model_directories(self) -> list[str]:
        return [
            str(Path(path).expanduser())
            for path in self.load_app_config().get("models", {}).get("directories", [])
        ]

    def get_llama_defaults(self) -> dict[str, Any]:
        return deepcopy(self.load_app_config()["defaults"]["llama"])

    def get_chat_defaults(self) -> dict[str, Any]:
        return deepcopy(self.load_app_config()["defaults"]["chat"])

    def get_launch_view(
        self,
        *,
        model_loaded: bool,
        models_available: bool,
        consume_first_launch: bool = False,
    ) -> str:
        config = self.load_app_config()
        if not config.get("first_launch_completed", False):
            view = "discover"
            if consume_first_launch:
                config["first_launch_completed"] = True
                self.save_app_config(config)
            return view

        if model_loaded:
            return "chat"
        if models_available:
            return "models"
        return "discover"

    def load_model_profiles(self) -> dict[str, Any]:
        self.ensure_initialized()
        profiles = self._read_json(self.model_profiles_file, {})
        if not isinstance(profiles, dict):
            profiles = {}
        profiles.setdefault("schema_version", SCHEMA_VERSION)
        profiles.setdefault("profiles", [])
        if not isinstance(profiles["profiles"], list):
            profiles["profiles"] = []
        return profiles

    def save_model_profiles(self, profiles: dict[str, Any]) -> None:
        profiles = {
            "schema_version": SCHEMA_VERSION,
            "profiles": profiles.get("profiles", []),
        }
        self._write_json(self.model_profiles_file, profiles)

    def get_model_settings_registry(self) -> dict[str, dict[str, Any]]:
        registry = {}
        for profile in self.load_model_profiles()["profiles"]:
            path = profile.get("path")
            if path:
                registry[path] = self.profile_to_settings(profile)
        return registry

    def get_model_profile(self, model_path: str) -> dict[str, Any] | None:
        target = self._resolve_path_string(model_path)
        for profile in self.load_model_profiles()["profiles"]:
            if self._paths_match(profile.get("path", ""), target):
                return deepcopy(profile)
        return None

    def get_model_profile_settings(self, model_path: str) -> dict[str, Any]:
        profile = self.get_model_profile(model_path)
        return self.profile_to_settings(profile) if profile else {}

    def save_model_profile(
        self,
        model_path: str,
        profile_settings: dict[str, Any],
        model_name: str | None = None,
    ) -> dict[str, Any]:
        profiles = self.load_model_profiles()
        now = datetime.now(UTC).isoformat()
        normalized_path = self._resolve_path_string(model_path)
        existing_index = None

        for idx, profile in enumerate(profiles["profiles"]):
            if self._paths_match(profile.get("path", ""), normalized_path):
                existing_index = idx
                break

        existing = profiles["profiles"][existing_index] if existing_index is not None else {}
        profile = self.settings_to_profile(
            model_path=normalized_path,
            profile_settings=profile_settings,
            model_name=model_name or existing.get("name"),
            existing=existing,
            updated_at=now,
        )

        if existing_index is None:
            profiles["profiles"].append(profile)
        else:
            profiles["profiles"][existing_index] = profile

        config = self.load_app_config()
        config["models"]["last_loaded"] = profile["id"]
        self.save_app_config(config)
        self.save_model_profiles(profiles)
        return deepcopy(profile)

    def profile_to_settings(self, profile: dict[str, Any] | None) -> dict[str, Any]:
        if not profile:
            return {}
        settings_payload = deepcopy(profile.get("load", {}))
        inference = deepcopy(profile.get("inference", {}))
        if inference:
            settings_payload["inference_settings"] = inference
        return settings_payload

    def settings_to_profile(
        self,
        *,
        model_path: str,
        profile_settings: dict[str, Any],
        model_name: str | None = None,
        existing: dict[str, Any] | None = None,
        updated_at: str | None = None,
    ) -> dict[str, Any]:
        existing = existing or {}
        model_file = Path(model_path)
        name = model_name or existing.get("name") or model_file.stem
        inference = deepcopy(profile_settings.get("inference_settings", {}))
        load_settings = {
            key: deepcopy(value)
            for key, value in profile_settings.items()
            if key != "inference_settings"
        }

        created_at = existing.get("created_at") or updated_at or datetime.now(UTC).isoformat()
        profile_id = existing.get("id") or self._profile_id(model_path, name)
        return {
            "id": profile_id,
            "name": name,
            "path": model_path,
            "fingerprint": self._fingerprint(model_file),
            "load": load_settings,
            "inference": inference,
            "created_at": created_at,
            "updated_at": updated_at or created_at,
        }

    def _migrate_legacy_model_settings(self) -> dict[str, Any]:
        profiles: list[dict[str, Any]] = []
        legacy = self._read_json(self.legacy_model_settings_file, {})
        if isinstance(legacy, dict):
            for model_path, profile_settings in legacy.items():
                if isinstance(profile_settings, dict):
                    profiles.append(
                        self.settings_to_profile(
                            model_path=self._resolve_path_string(model_path),
                            profile_settings=profile_settings,
                        )
                    )
        return {"schema_version": SCHEMA_VERSION, "profiles": profiles}

    def _default_config(self, workspace_root: str | Path | None = None) -> dict[str, Any]:
        root = Path(workspace_root or settings.WORKSPACE_ROOT or Path.cwd()).expanduser()
        return {
            "schema_version": SCHEMA_VERSION,
            "first_launch_completed": False,
            "workspace": {
                "root": str(root),
                "disable_sandbox": bool(settings.DISABLE_SANDBOX),
                "command_policy": "allow",
            },
            "models": {
                "directories": list(settings.MODEL_DIRS),
                "last_loaded": "",
            },
            "defaults": {
                "llama": {
                    "ctx_size": settings.LLAMA_SERVER_CTX_SIZE,
                    "gpu_layers": settings.LLAMA_SERVER_GPU_LAYERS,
                    "flash_attn": settings.LLAMA_SERVER_FLASH_ATTN,
                    "kv_cache_type": settings.LLAMA_SERVER_KV_CACHE_TYPE,
                    "vocab_type": settings.LLAMA_SERVER_VOCAB_TYPE,
                    "task_timeout": settings.LLAMA_SERVER_TASK_TIMEOUT,
                },
                "chat": {
                    "system_prompt": settings.DEFAULT_SYSTEM_PROMPT,
                    "temperature": settings.DEFAULT_TEMPERATURE,
                    "top_p": settings.DEFAULT_TOP_P,
                    "max_tokens": settings.DEFAULT_MAX_TOKENS,
                    "request_timeout": settings.CHAT_REQUEST_TIMEOUT,
                    "max_tool_iterations": settings.MAX_TOOL_ITERATIONS,
                },
            },
        }

    def _merge_app_config(self, config: dict[str, Any]) -> dict[str, Any]:
        merged = self._default_config()
        self._deep_update(merged, config if isinstance(config, dict) else {})
        merged["schema_version"] = SCHEMA_VERSION
        return merged

    def _read_json(self, path: Path, fallback: Any) -> Any:
        if not path.exists():
            return deepcopy(fallback)
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as exc:
            logger.error("[config] Could not read %s: %s", path, exc)
            return deepcopy(fallback)

    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        shutil.move(str(tmp_path), str(path))

    def _resolve_path_string(self, path: str) -> str:
        try:
            return str(Path(path).expanduser().resolve())
        except OSError:
            return str(Path(path).expanduser())

    def _paths_match(self, left: str, right: str) -> bool:
        if not left or not right:
            return False
        return self._resolve_path_string(left) == self._resolve_path_string(right)

    def _fingerprint(self, model_file: Path) -> dict[str, Any]:
        fingerprint: dict[str, Any] = {"filename": model_file.name}
        try:
            stat = model_file.stat()
            fingerprint["size"] = stat.st_size
        except OSError:
            fingerprint["size"] = None
        return fingerprint

    def _profile_id(self, model_path: str, name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "model"
        digest = hashlib.sha1(model_path.encode("utf-8")).hexdigest()[:10]
        return f"{slug}-{digest}"

    def _deep_update(self, base: dict[str, Any], updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value


config_loader = ConfigLoader()
