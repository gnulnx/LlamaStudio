"""Managed local speech-to-text through the whisper.cpp HTTP server."""

from __future__ import annotations

import hashlib
import io
import os
import platform
import shutil
import socket
import subprocess
import tarfile
import tempfile
import threading
import time
import wave
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import httpx

from .config import settings
from .config_store import config_loader
from .logger import logger
from .tools import check_path_safe

WHISPER_VERSION = "v1.9.1"
WHISPER_RELEASE_URL = f"https://github.com/ggml-org/whisper.cpp/releases/download/{WHISPER_VERSION}"
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

WHISPER_ASSETS = {
    ("Linux", "x86_64"): (
        "whisper-bin-ubuntu-x64.tar.gz",
        "f3bf3b4369a99b54665b0f19b88483b30de27f25963b0414235dea03198515c5",
    ),
    ("Linux", "aarch64"): (
        "whisper-bin-ubuntu-arm64.tar.gz",
        "e0b66cd551ff6f2a28fabe3c6e89691eea037bb76833493abb9a71ca788994b3",
    ),
}

WHISPER_MODELS = {
    "base.en": (
        147_964_211,
        "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002",
    ),
    "small.en": (
        487_614_201,
        "c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d",
    ),
    "large-v3-turbo-q5_0": (
        574_041_195,
        "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
    ),
}

ProgressCallback = Callable[[int, int | None, str], None]


class SpeechError(RuntimeError):
    """A user-actionable speech engine failure."""


class SpeechManager:
    """Own whisper.cpp installation, process lifecycle, and transcription."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._current_model: str | None = None
        self._recorder_process: subprocess.Popen | None = None
        self._recorder_path: Path | None = None
        self._lock = threading.RLock()

    @staticmethod
    def _install_root(config: dict | None = None) -> Path:
        speech_config = config or config_loader.get_speech_config()
        return Path(speech_config.get("install_dir") or settings.SPEECH_DIR).expanduser()

    @staticmethod
    def _model_path(model: str, install_root: Path) -> Path:
        return install_root / "models" / f"ggml-{model}.bin"

    def resolve_binary(
        self,
        install_root: Path | None = None,
        *,
        include_path: bool = True,
    ) -> Path | None:
        configured = os.environ.get("LLAMASTUDIO_WHISPER_SERVER_BIN")
        if configured:
            candidate = Path(configured).expanduser()
            if candidate.is_file():
                return candidate.resolve()

        root = install_root or self._install_root()
        preferred = root / "bin" / WHISPER_VERSION
        if preferred.exists():
            candidates = sorted(preferred.glob("*/whisper-server"))
            if candidates:
                return candidates[0].resolve()

        if not include_path:
            return None
        resolved = shutil.which("whisper-server")
        return Path(resolved).resolve() if resolved else None

    @staticmethod
    def _download(
        url: str,
        destination: Path,
        expected_sha256: str,
        expected_size: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(f"{destination.suffix}.part")
        digest = hashlib.sha256()
        written = 0
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
            response.raise_for_status()
            total = expected_size or int(response.headers.get("content-length", 0)) or None
            with open(partial, "wb") as output:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    if progress:
                        progress(written, total, destination.name)

        if expected_size is not None and written != expected_size:
            partial.unlink(missing_ok=True)
            raise SpeechError(
                f"Download size mismatch for {destination.name}: expected "
                f"{expected_size}, received {written} bytes."
            )
        if digest.hexdigest() != expected_sha256:
            partial.unlink(missing_ok=True)
            raise SpeechError(f"Checksum verification failed for {destination.name}.")
        partial.replace(destination)

    @staticmethod
    def _extract_release(archive: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        destination_root = destination.resolve()
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                target = (destination / member.name).resolve()
                if destination_root != target and destination_root not in target.parents:
                    raise SpeechError("The whisper.cpp release archive contains an unsafe path.")
                if member.issym():
                    link_target = (target.parent / member.linkname).resolve()
                    if (
                        link_target != destination_root
                        and destination_root not in link_target.parents
                    ):
                        raise SpeechError(
                            "The whisper.cpp release archive contains an unsafe link."
                        )
                elif member.islnk():
                    link_target = (destination / member.linkname).resolve()
                    if (
                        link_target != destination_root
                        and destination_root not in link_target.parents
                    ):
                        raise SpeechError(
                            "The whisper.cpp release archive contains an unsafe link."
                        )
            bundle.extractall(destination)

    @staticmethod
    def _file_matches(
        path: Path,
        expected_sha256: str,
        expected_size: int | None = None,
    ) -> bool:
        if not path.is_file():
            return False
        if expected_size is not None and path.stat().st_size != expected_size:
            return False
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == expected_sha256

    def install(
        self,
        model: str = "small.en",
        install_dir: str | Path | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict:
        """Install a pinned whisper.cpp binary and verified model."""
        if model not in WHISPER_MODELS:
            choices = ", ".join(WHISPER_MODELS)
            raise SpeechError(f"Unknown Whisper model '{model}'. Choose one of: {choices}.")

        machine = platform.machine().lower()
        machine = "x86_64" if machine in {"amd64", "x64"} else machine
        asset = WHISPER_ASSETS.get((platform.system(), machine))
        if asset is None:
            raise SpeechError(
                f"No prebuilt whisper.cpp {WHISPER_VERSION} release is available for "
                f"{platform.system()} {platform.machine()}."
            )

        root = (
            Path(install_dir).expanduser().resolve()
            if install_dir
            else self._install_root().resolve()
        )
        archive_name, archive_sha256 = asset
        release_dir = root / "bin" / WHISPER_VERSION
        binary = self.resolve_binary(root, include_path=False)
        if binary is None:
            archive = root / "downloads" / archive_name
            if not self._file_matches(archive, archive_sha256):
                archive.unlink(missing_ok=True)
                self._download(
                    f"{WHISPER_RELEASE_URL}/{archive_name}",
                    archive,
                    archive_sha256,
                    progress=progress,
                )
            self._extract_release(archive, release_dir)
            binary = self.resolve_binary(root, include_path=False)
            if binary is None:
                raise SpeechError("The downloaded release did not contain whisper-server.")

        model_path = self._model_path(model, root)
        expected_size, expected_sha256 = WHISPER_MODELS[model]
        if not self._file_matches(model_path, expected_sha256, expected_size):
            model_path.unlink(missing_ok=True)
            self._download(
                f"{MODEL_URL}/{model_path.name}",
                model_path,
                expected_sha256,
                expected_size,
                progress,
            )

        speech_config = config_loader.get_speech_config()
        speech_config.update({"install_dir": str(root), "model": model})
        config_loader.save_speech_config(speech_config)
        return {
            "binary": str(binary),
            "model": model,
            "model_path": str(model_path),
            "install_dir": str(root),
        }

    @staticmethod
    def _port_in_use() -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            return sock.connect_ex(("127.0.0.1", settings.SPEECH_SERVER_PORT)) == 0

    @property
    def is_running(self) -> bool:
        if self._process is not None and self._process.poll() is not None:
            self._process = None
            self._current_model = None
        if not self._port_in_use():
            return False
        if self._process is not None:
            return True
        try:
            response = httpx.get(
                f"http://127.0.0.1:{settings.SPEECH_SERVER_PORT}/",
                timeout=1.0,
            )
            return response.status_code == 200 and "Whisper.cpp Server" in response.text
        except httpx.HTTPError:
            return False

    def _wait_for_ready(self, timeout: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{settings.SPEECH_SERVER_PORT}/"
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                return False
            try:
                if httpx.get(url, timeout=1.0).status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        return False

    @staticmethod
    def _log_path() -> Path:
        path = Path(settings.LOG_DIR).expanduser() / "speech.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def start(self, model: str | None = None, use_gpu: bool | None = None) -> dict:
        """Start whisper-server, reusing it across microphone turns."""
        with self._lock:
            speech_config = config_loader.get_speech_config()
            selected_model = model or str(speech_config.get("model") or "small.en")
            root = self._install_root(speech_config)
            binary = self.resolve_binary(root)
            model_path = self._model_path(selected_model, root)

            if binary is None or not model_path.is_file():
                raise SpeechError(
                    "Whisper is not installed. Run "
                    f"'lls speech install --model {selected_model}' first."
                )
            if self.is_running:
                if self._current_model and self._current_model != selected_model:
                    self.stop()
                else:
                    return self.get_status()
            elif self._port_in_use():
                raise SpeechError(
                    f"Port {settings.SPEECH_SERVER_PORT} is already in use by another service."
                )

            gpu_enabled = bool(speech_config.get("use_gpu", False))
            if use_gpu is not None:
                gpu_enabled = use_gpu
            command = [
                str(binary),
                "--model",
                str(model_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(settings.SPEECH_SERVER_PORT),
            ]
            if not gpu_enabled:
                command.append("--no-gpu")

            environment = os.environ.copy()
            library_path = str(binary.parent)
            if environment.get("LD_LIBRARY_PATH"):
                library_path += os.pathsep + environment["LD_LIBRARY_PATH"]
            environment["LD_LIBRARY_PATH"] = library_path

            log_path = self._log_path()
            logger.info("[speech] Starting whisper.cpp model %s", selected_model)
            with open(log_path, "w") as log_file:
                self._process = subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=binary.parent,
                    env=environment,
                    start_new_session=True,
                )
            self._current_model = selected_model

            if not self._wait_for_ready():
                self.stop()
                detail = ""
                with suppress(OSError):
                    detail = "\n".join(log_path.read_text(errors="replace").splitlines()[-8:])
                raise SpeechError(
                    "whisper-server did not become ready. Check speech.log."
                    + (f"\n{detail}" if detail else "")
                )

            speech_config.update({"model": selected_model, "use_gpu": gpu_enabled})
            config_loader.save_speech_config(speech_config)
            return self.get_status()

    def stop(self) -> bool:
        """Stop only the whisper-server process owned by LlamaStudio."""
        self.cancel_system_recording()
        with self._lock:
            process = self._process
            self._process = None
            self._current_model = None
            if process is None or process.poll() is not None:
                return False
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            return True

    @property
    def is_system_recording(self) -> bool:
        process = self._recorder_process
        return process is not None and process.poll() is None

    def start_system_recording(self) -> dict:
        """Record the OS default PipeWire/Pulse input when the browser cannot."""
        with self._lock:
            if self.is_system_recording:
                return {"recording": True, "capture": "system"}

            recorder = shutil.which("arecord")
            if recorder is None:
                raise SpeechError("arecord is required for system microphone capture.")
            workspace = check_path_safe(".")
            with tempfile.NamedTemporaryFile(
                prefix=".lls-browser-speech-",
                suffix=".wav",
                dir=workspace,
                delete=False,
            ) as temporary:
                recording_path = Path(temporary.name)

            process = subprocess.Popen(
                [
                    recorder,
                    "--quiet",
                    "--device",
                    "pulse",
                    "--format",
                    "S16_LE",
                    "--rate",
                    "16000",
                    "--channels",
                    "1",
                    "--file-type",
                    "wav",
                    str(recording_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            time.sleep(0.15)
            if process.poll() is not None:
                detail = (process.stderr.read() if process.stderr else b"").decode(
                    "utf-8", errors="replace"
                )
                recording_path.unlink(missing_ok=True)
                raise SpeechError(detail.strip() or "The system microphone could not be opened.")

            self._recorder_process = process
            self._recorder_path = recording_path
            logger.info("[speech] Recording from the OS default microphone")
            return {"recording": True, "capture": "system"}

    def cancel_system_recording(self) -> bool:
        """Stop and discard an active system microphone recording."""
        with self._lock:
            process = self._recorder_process
            recording_path = self._recorder_path
            self._recorder_process = None
            self._recorder_path = None
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            if recording_path is not None:
                recording_path.unlink(missing_ok=True)
            return process is not None

    def stop_system_recording(
        self,
        *,
        language: str = "auto",
        translate: bool = False,
    ) -> dict:
        """Stop the OS recorder and transcribe its captured WAV."""
        with self._lock:
            process = self._recorder_process
            recording_path = self._recorder_path
            self._recorder_process = None
            self._recorder_path = None
            if process is None or recording_path is None:
                raise SpeechError("No system microphone recording is active.")
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)

            try:
                if not recording_path.is_file() or recording_path.stat().st_size <= 44:
                    detail = (process.stderr.read() if process.stderr else b"").decode(
                        "utf-8", errors="replace"
                    )
                    raise SpeechError(detail.strip() or "The microphone recording was empty.")
                audio = recording_path.read_bytes()
            finally:
                recording_path.unlink(missing_ok=True)

        return self.transcribe_bytes(
            audio,
            language=language,
            translate=translate,
        )

    def get_status(self) -> dict:
        speech_config = config_loader.get_speech_config()
        root = self._install_root(speech_config)
        model = self._current_model or str(speech_config.get("model") or "small.en")
        binary = self.resolve_binary(root)
        model_path = self._model_path(model, root)
        return {
            "engine": "whisper.cpp",
            "version": WHISPER_VERSION,
            "installed": binary is not None,
            "binary": str(binary) if binary else None,
            "running": self.is_running,
            "managed": self._process is not None,
            "model": model,
            "model_path": str(model_path),
            "model_installed": model_path.is_file(),
            "language": speech_config.get("language", "auto"),
            "use_gpu": bool(speech_config.get("use_gpu", False)),
            "system_recording": self.is_system_recording,
            "port": settings.SPEECH_SERVER_PORT,
        }

    @staticmethod
    def _normalize_audio(audio: bytes) -> bytes:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise SpeechError("ffmpeg is required to decode microphone audio.")
        try:
            result = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    "pipe:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    "-f",
                    "s16le",
                    "pipe:1",
                ],
                input=audio,
                capture_output=True,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SpeechError("Audio conversion timed out.") from exc
        if result.returncode != 0 or not result.stdout:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            logger.warning("[speech] Audio conversion failed: %s", detail)
            raise SpeechError("The recording could not be decoded.")
        wav = io.BytesIO()
        with wave.open(wav, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16_000)
            output.writeframes(result.stdout)
        return wav.getvalue()

    def transcribe_bytes(
        self,
        audio: bytes,
        *,
        language: str = "auto",
        translate: bool = False,
    ) -> dict:
        if not audio:
            raise SpeechError("The recording was empty.")
        if len(audio) > settings.SPEECH_MAX_AUDIO_BYTES:
            raise SpeechError("The recording exceeds the 100 MB limit.")

        with self._lock:
            if not self.is_running:
                self.start()
            wav = self._normalize_audio(audio)
            data = {
                "temperature": "0.0",
                "response_format": "json",
                "translate": "true" if translate else "false",
            }
            if language and language != "auto":
                data["language"] = language
            try:
                response = httpx.post(
                    f"http://127.0.0.1:{settings.SPEECH_SERVER_PORT}/inference",
                    files={"file": ("recording.wav", wav, "audio/wav")},
                    data=data,
                    timeout=300.0,
                )
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise SpeechError(f"Whisper transcription failed: {exc}") from exc

        text = str(payload.get("text", "")).strip()
        if not text:
            raise SpeechError("Whisper did not detect any speech.")
        return {"text": text, "engine": "whisper.cpp", "model": self._current_model}

    def transcribe_file(
        self,
        audio_path: str,
        *,
        language: str = "auto",
        translate: bool = False,
    ) -> dict:
        safe_path = check_path_safe(audio_path)
        if not safe_path.is_file():
            raise SpeechError(f"Audio file not found: {audio_path}")
        return self.transcribe_bytes(
            safe_path.read_bytes(),
            language=language,
            translate=translate,
        )


speech = SpeechManager()
