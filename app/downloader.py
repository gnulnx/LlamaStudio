import os
import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import httpx
from .config import settings
from .logger import logger

class ModelDownloader:
    _instance = None
    _active_task: Optional[asyncio.Task] = None
    _cancel_event: Optional[asyncio.Event] = None
    
    # Progress state variables
    repo_id: Optional[str] = None
    filename: Optional[str] = None
    downloaded_bytes: int = 0
    total_bytes: int = 0
    start_time: float = 0.0
    status: str = "idle"  # idle, downloading, completed, failed, cancelled
    error_message: Optional[str] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def is_active(self) -> bool:
        return self._active_task is not None and not self._active_task.done()

    def get_progress(self) -> Dict[str, Any]:
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
            "error": self.error_message
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
            try:
                await self._active_task
            except asyncio.CancelledError:
                pass
        self._active_task = None
        self._cancel_event = None

    async def _download_loop(self, repo_id: str, filename: str):
        # Build local target directory: settings.MODEL_DIRS[0] / author / repo_name / filename
        base_dir = Path(settings.MODEL_DIRS[0] if settings.MODEL_DIRS else "/home/gnulnx/.lmstudio/models")
        
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
            
            download_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            logger.info(f"[downloader] Fetching GGUF from: {download_url}")
            
            headers = {"User-Agent": "LLamaStudio-Client"}
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("GET", download_url, headers=headers, follow_redirects=True) as response:
                    if response.status_code != 200:
                        raise ValueError(f"HTTP Error {response.status_code} from Hugging Face hub.")
                        
                    self.total_bytes = int(response.headers.get("content-length", 0))
                    self.start_time = time.time()
                    
                    with open(tmp_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                            if self._cancel_event and self._cancel_event.is_set():
                                logger.info("[downloader] Download cancelled during write chunk loop.")
                                return
                                
                            f.write(chunk)
                            self.downloaded_bytes += len(chunk)
                            
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
        except Exception as e:
            self.status = "failed"
            self.error_message = str(e)
            logger.error(f"[downloader] Download failed: {e}", exc_info=True)
            self._cleanup_temp_file(tmp_path)
        finally:
            self._active_task = None
            self._cancel_event = None

    def _cleanup_temp_file(self, tmp_path: Path):
        try:
            if tmp_path.exists():
                tmp_path.unlink()
                logger.info(f"[downloader] Cleaned up temporary file: {tmp_path}")
        except Exception as e:
            logger.error(f"[downloader] Failed to clean up temp file '{tmp_path}': {e}")

downloader = ModelDownloader()
