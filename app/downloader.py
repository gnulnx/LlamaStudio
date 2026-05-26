import asyncio
import contextlib
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from .config import settings
from .logger import logger


class _DownloadCancelledError(Exception):
    """Raised when the user cancels an active transfer."""


class _NonRetryableDownloadError(Exception):
    """Raised for Hub responses that another retry cannot fix."""


class ModelDownloader:
    _instance = None
    _active_task: asyncio.Task | None = None
    _cancel_event: asyncio.Event | None = None

    # Progress state variables
    repo_id: str | None = None
    filename: str | None = None
    downloaded_bytes: int = 0
    total_bytes: int = 0
    start_time: float = 0.0
    status: str = "idle"  # idle, downloading, completed, failed, cancelled
    error_message: str | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def is_active(self) -> bool:
        return self._active_task is not None and not self._active_task.done()

    def get_progress(self) -> dict[str, Any]:
        if self.status == "idle":
            return {"status": "idle"}

        elapsed = time.time() - self.start_time if self.start_time > 0 else 0
        speed = self.downloaded_bytes / elapsed if elapsed > 0 else 0.0
        percent = (self.downloaded_bytes / self.total_bytes * 100) if self.total_bytes > 0 else 0.0
        eta = (self.total_bytes - self.downloaded_bytes) / speed if speed > 0 else 0.0

        return {
            "repo_id": self.repo_id,
            "filename": self.filename,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "percent": round(percent, 2),
            "speed_mb": round(speed / (1024 * 1024), 2),  # MB/s
            "eta_seconds": int(eta) if self.total_bytes > 0 and speed > 0 else 0,
            "status": self.status,
            "error": self.error_message,
        }

    async def start_download(self, repo_id: str, filename: str) -> bool:
        if self.is_active:
            logger.warning("[downloader] Another download task is already active.")
            return False

        self.repo_id = repo_id
        self.filename = filename
        self.downloaded_bytes = 0
        self.total_bytes = 0
        self.start_time = 0.0
        self.status = "downloading"
        self.error_message = None

        self._cancel_event = asyncio.Event()
        self._active_task = asyncio.create_task(self._download_loop(repo_id, filename))
        return True

    async def cancel_download(self):
        if not self.is_active:
            return

        logger.info("[downloader] Cancelling active download task...")
        self.status = "cancelled"
        if self._cancel_event:
            self._cancel_event.set()
        if self._active_task:
            self._active_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._active_task
        self._active_task = None
        self._cancel_event = None

    async def _download_loop(self, repo_id: str, filename: str):
        # Build local target directory: settings.MODEL_DIRS[0] / author / repo_name / filename
        base_dir = Path(
            settings.MODEL_DIRS[0]
            if settings.MODEL_DIRS
            else str(Path.home() / ".lmstudio" / "models")
        )

        parts = repo_id.split("/")
        if len(parts) > 1:
            author, repo_name = parts[0], parts[1]
        else:
            author, repo_name = "huggingface", parts[0]

        target_dir = base_dir / author / repo_name
        target_path = target_dir / filename
        tmp_path = target_dir / f"{filename}.tmp"

        try:
            target_dir.mkdir(parents=True, exist_ok=True)

            await self._download_from_huggingface(
                repo_id,
                filename,
                tmp_path,
            )

            if self._cancel_event and self._cancel_event.is_set():
                return

            if tmp_path.exists():
                tmp_path.rename(target_path)

            self.status = "completed"
            logger.info(f"[downloader] Download completed successfully: {target_path}")

            # Trigger model list cache refresh
            from .model_manager import refresh_models

            refresh_models()

        except asyncio.CancelledError:
            self.status = "cancelled"
            logger.info("[downloader] Download task cancelled asynchronously.")
            self._cleanup_temp_file(tmp_path)
        except _DownloadCancelledError:
            self.status = "cancelled"
            logger.info("[downloader] Download cancelled during write chunk loop.")
            self._cleanup_temp_file(tmp_path)
        except Exception as e:
            self.status = "failed"
            self.error_message = str(e)
            logger.error(f"[downloader] Download failed: {e}", exc_info=True)
            self._cleanup_temp_file(tmp_path)
        finally:
            self._active_task = None
            self._cancel_event = None

    async def _download_from_huggingface(self, repo_id: str, filename: str, tmp_path: Path):
        download_url = self._resolve_url(repo_id, filename)
        logger.info(f"[downloader] Fetching GGUF from: {download_url}")

        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        existing_bytes = tmp_path.stat().st_size if tmp_path.exists() else 0
        self.downloaded_bytes = existing_bytes
        if self.start_time <= 0:
            self.start_time = time.time()

        max_attempts = 6
        attempts = 0

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=300.0)) as client:
            while True:
                if self._cancel_event and self._cancel_event.is_set():
                    raise _DownloadCancelledError()

                headers = self._hf_headers()
                headers["Range"] = f"bytes={self.downloaded_bytes}-"

                try:
                    async with client.stream(
                        "GET", download_url, headers=headers, follow_redirects=True
                    ) as response:
                        if (
                            response.status_code == 416
                            and self.total_bytes
                            and self.downloaded_bytes >= self.total_bytes
                        ):
                            return

                        if response.status_code not in (200, 206):
                            body = await response.aread()
                            detail = body[:300].decode("utf-8", errors="replace").strip()
                            message = (
                                f"HTTP Error {response.status_code} from Hugging Face hub."
                                + (f" {detail}" if detail else "")
                            )
                            if 400 <= response.status_code < 500:
                                raise _NonRetryableDownloadError(message)
                            raise ValueError(message)

                        if response.status_code == 200 and self.downloaded_bytes > 0:
                            logger.info(
                                "[downloader] Server ignored resume range; restarting temp download."
                            )
                            self.downloaded_bytes = 0
                            tmp_path.write_bytes(b"")

                        self._update_total_bytes(response)
                        mode = (
                            "ab"
                            if response.status_code == 206 and self.downloaded_bytes > 0
                            else "wb"
                        )

                        with open(tmp_path, mode) as f:
                            async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                                if self._cancel_event and self._cancel_event.is_set():
                                    raise _DownloadCancelledError()

                                f.write(chunk)
                                self.downloaded_bytes += len(chunk)

                        if self.total_bytes <= 0 or self.downloaded_bytes >= self.total_bytes:
                            return

                        attempts += 1
                        if attempts >= max_attempts:
                            raise ValueError(
                                f"Hugging Face download ended early after {self.downloaded_bytes} "
                                f"of {self.total_bytes} bytes."
                            )
                        logger.warning(
                            "[downloader] Hugging Face stream ended early; resuming from byte %s.",
                            self.downloaded_bytes,
                        )

                except _DownloadCancelledError:
                    raise
                except _NonRetryableDownloadError:
                    raise
                except Exception:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise
                    logger.warning(
                        "[downloader] Hugging Face transfer failed; retrying from byte %s.",
                        self.downloaded_bytes,
                        exc_info=True,
                    )
                    await asyncio.sleep(min(2**attempts, 10))

    def _hf_headers(self) -> dict[str, str]:
        headers = {"User-Agent": "LLamaStudio-Client"}
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _resolve_url(self, repo_id: str, filename: str) -> str:
        quoted_repo = quote(repo_id.strip("/"), safe="/")
        quoted_filename = quote(filename.lstrip("/"), safe="/")
        return f"https://huggingface.co/{quoted_repo}/resolve/main/{quoted_filename}"

    def _update_total_bytes(self, response: httpx.Response):
        content_range = response.headers.get("content-range", "")
        if "/" in content_range:
            total_part = content_range.rsplit("/", 1)[-1]
            if total_part.isdigit():
                self.total_bytes = int(total_part)
                return

        content_length = response.headers.get("content-length")
        if content_length and content_length.isdigit():
            self.total_bytes = self.downloaded_bytes + int(content_length)

    def _cleanup_temp_file(self, tmp_path: Path):
        try:
            if tmp_path.exists():
                tmp_path.unlink()
                logger.info(f"[downloader] Cleaned up temporary file: {tmp_path}")
        except Exception as e:
            logger.error(f"[downloader] Failed to clean up temp file '{tmp_path}': {e}")


downloader = ModelDownloader()
