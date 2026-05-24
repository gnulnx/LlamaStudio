"""
Manages the llama-server process lifecycle.
The app runs independently - llama-server is only running when a model is loaded.
"""

import socket
import subprocess
import time
from pathlib import Path
from typing import ClassVar

from .config import settings
from .logger import logger


class ServerManager:
    _instance = None
    _process: subprocess.Popen | None = None
    _current_model: str | None = None
    _current_model_name: str | None = None
    _current_params: ClassVar[dict] = {}
    _is_loading: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _port_in_use(self) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect(("127.0.0.1", settings.LLAMA_SERVER_PORT))
            sock.close()
            return True
        except (ConnectionRefusedError, OSError):
            return False

    @property
    def is_running(self) -> bool:
        # Check if port is in use and process is active (if we started it)
        port_active = self._port_in_use()
        if not port_active and self._process is not None:
            # Process crashed or stopped
            self._process = None
            self._current_model = None
            self._current_model_name = None
        return port_active

    def _wait_for_ready(self, timeout: int = 180) -> bool:
        import json
        import urllib.request

        start = time.time()
        url = f"http://127.0.0.1:{settings.LLAMA_SERVER_PORT}/health"

        while time.time() - start < timeout:
            # If our process died while loading, stop waiting
            if self._process is not None and self._process.poll() is not None:
                logger.error(
                    f"[server] Process terminated unexpectedly with code {self._process.returncode}"
                )
                return False
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=1) as response:
                    if response.status == 200:
                        data = json.loads(response.read().decode())
                        # llama-server /health returns status="ok" or connected
                        if data.get("status") in ["ok", "connected"]:
                            return True
            except Exception:
                # Expect connection refused, timeouts, or 503 Service Unavailable during startup
                pass
            time.sleep(1)
        return False

    def _build_command(
        self, model_path: str, params: dict | None = None, force_cpu: bool = False
    ) -> list:
        if params is None:
            params = {}

        # Merge input params with default settings
        ctx_size = params.get("ctx_size", settings.LLAMA_SERVER_CTX_SIZE)
        gpu_layers = 0 if force_cpu else params.get("gpu_layers", settings.LLAMA_SERVER_GPU_LAYERS)
        flash_attn = params.get("flash_attn", settings.LLAMA_SERVER_FLASH_ATTN)
        kv_cache_type = params.get("kv_cache_type", settings.LLAMA_SERVER_KV_CACHE_TYPE)
        vocab_type = params.get("vocab_type", settings.LLAMA_SERVER_VOCAB_TYPE)
        threads = params.get("threads", 4)
        mmap = params.get("mmap", True)
        seed = params.get("seed", -1)
        rope_freq_base = params.get("rope_freq_base", 0.0)
        rope_freq_scale = params.get("rope_freq_scale", 0.0)

        from .config import resolve_llama_server_bin

        binary_path = resolve_llama_server_bin(force_cpu=force_cpu)

        cmd = [
            binary_path,
            "-m",
            model_path,
            "--port",
            str(settings.LLAMA_SERVER_PORT),
            "--ctx-size",
            str(ctx_size),
            "--gpu-layers",
            str(gpu_layers),
            "-ctk",
            str(kv_cache_type),
            "-ctv",
            str(vocab_type),
        ]

        # Handle flash-attn
        if flash_attn is True or flash_attn == "on":
            cmd.extend(["--flash-attn", "on"])
        else:
            cmd.extend(["--flash-attn", "off"])

        # Handle threads
        if threads and int(threads) > 0:
            cmd.extend(["--threads", str(threads)])

        # Handle mmap
        if not mmap:
            cmd.append("--no-mmap")
        else:
            cmd.append("--mmap")

        # Handle seed
        if seed is not None and int(seed) >= 0:
            cmd.extend(["--seed", str(seed)])

        # Handle Rope Frequency Base and Scale
        if rope_freq_base and float(rope_freq_base) > 0:
            cmd.extend(["--rope-freq-base", str(rope_freq_base)])
        if rope_freq_scale and float(rope_freq_scale) > 0:
            cmd.extend(["--rope-freq-scale", str(rope_freq_scale)])

        # Add override-kv if specified in model params
        override_kv = params.get("override_kv")
        if override_kv:
            cmd.extend(["--override-kv", override_kv])
        elif "qwen" in model_path.lower():
            # Only apply Qwen override fallback to Qwen models
            cmd.extend(["--override-kv", "qwen35.context_length=int:262144"])

        # Auto-configure deepseek models (ensure reasoning format is correct)
        if "deepseek" in model_path.lower():
            cmd.extend(["--reasoning-format", "deepseek"])

        # Add chat template if specified
        chat_template = params.get("chat_template")
        custom_template = params.get("custom_template")
        if chat_template == "custom" and custom_template:
            cmd.extend(["--chat-template", custom_template])
        elif chat_template == "deepseek-r1":
            templates_dir = Path.home() / "llama.cpp" / "models" / "templates"
            generic_r1_tmpl = templates_dir / "llama-cpp-deepseek-r1.jinja"
            specific_qwen_tmpl = templates_dir / "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B.jinja"

            if generic_r1_tmpl.exists():
                cmd.extend(["--chat-template-file", str(generic_r1_tmpl)])
            elif specific_qwen_tmpl.exists():
                cmd.extend(["--chat-template-file", str(specific_qwen_tmpl)])
            else:
                cmd.extend(["--chat-template", "deepseek"])
        elif chat_template == "chatml":
            chatml_tmpl = "{% for message in messages %}{{'</think>' + message['role'] + '\\n' + message['content'] + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{'\\n'}}{% endif %}"
            cmd.extend(["--chat-template", chatml_tmpl])
        elif chat_template == "gemma":
            gemma_tmpl = "{% for message in messages %}{{'<start_of_turn>' + message['role'] + '\\n' + message['content'] + '<end_of_turn>\\n'}}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>assistant\\n'}}{% endif %}"
            cmd.extend(["--chat-template", gemma_tmpl])
        elif chat_template == "llama3":
            llama3_tmpl = "<|begin_of_text|>{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>'}}{% endfor %}{% if add_generation_prompt %}{{'<|start_header_id|>assistant<|end_header_id|>\\n\\n'}}{% endif %}"
            cmd.extend(["--chat-template", llama3_tmpl])

        return cmd

    def _write_log(self) -> Path:
        log_dir = Path(settings.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / "server.log"

    def load_model(self, model_path: str, params: dict | None = None) -> bool:
        """Load a model. If one is already loaded, eject it first."""
        model = model_path
        log_file = self._write_log()
        params = params or {}

        # Determine initial cpu mode from params
        cpu_mode = params.get("cpu_mode", False) or int(params.get("gpu_layers", 999)) == 0

        try:
            if not Path(model).exists():
                error_msg = f"[server] Model not found: {model}"
                logger.error(error_msg)
                with open(log_file, "a") as f:
                    f.write(f"\nERROR: {error_msg}\n")
                return False

            if self.is_running:
                logger.info("[server] Ejecting current model first...")
                self.eject_model()

            self._current_model = model
            self._current_model_name = Path(model).stem
            self._current_params = params
            self._is_loading = True

            cmd = self._build_command(model, params, force_cpu=cpu_mode)

            logger.info(f"[server] Loading: {model} with cmd: {' '.join(cmd)}")
            with open(log_file, "w") as f:
                f.write("--- LLamaStudio Server Starting ---\n")
                f.write(f"Model Path: {model}\n")
                f.write(f"Command: {' '.join(cmd)}\n\n")
                f.flush()
                self._process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=Path(cmd[0]).parent,
                )

            if self._wait_for_ready():
                logger.info(f"[server] Model loaded: {self._current_model_name}")
                self._is_loading = False
                return True

            # AUTOMATIC FALLBACK LOGIC
            # If startup failed and we were NOT in CPU mode, try starting in CPU Mode as a fallback!
            if not cpu_mode:
                logger.warning(
                    "[server] GPU loading failed or timed out. Attempting automatic CPU fallback..."
                )
                with open(log_file, "a") as f:
                    f.write("\n[WARNING] GPU load failed. Triggering automatic CPU Fallback...\n\n")

                self.eject_model()
                self._current_model = model
                self._current_model_name = Path(model).stem
                self._current_params = params
                self._is_loading = True

                # Build fallback command with force_cpu=True
                cmd_fallback = self._build_command(model, params, force_cpu=True)
                logger.info(
                    f"[server] Loading fallback: {model} with cmd: {' '.join(cmd_fallback)}"
                )

                with open(log_file, "a") as f:
                    f.write("--- Triggering CPU Fallback Server ---\n")
                    f.write(f"Command: {' '.join(cmd_fallback)}\n\n")
                    f.flush()

                self._process = subprocess.Popen(
                    cmd_fallback,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=Path(cmd_fallback[0]).parent,
                )

                if self._wait_for_ready():
                    logger.info(
                        f"[server] Model loaded successfully via CPU Fallback: {self._current_model_name}"
                    )
                    # Update parameters to reflect CPU mode was used
                    self._current_params["cpu_mode"] = True
                    self._current_params["gpu_layers"] = 0
                    self._is_loading = False
                    return True

            # If we get here, both GPU and fallback (or CPU alone) failed
            error_msg = "[server] Timed out waiting for llama-server to become ready."
            logger.error(error_msg)
            with open(log_file, "a") as f:
                f.write(f"\nERROR: {error_msg}\n")
                f.write("DEBUG INFO:\n")
                f.write("- Port 1234 might be blocked or in use by another zombie process.\n")
                f.write(
                    "- Model configuration parameters (e.g. context size, thread count) might exceed system capability.\n"
                )
                f.write(
                    "- GPU out-of-memory: Check if the GGUF model is too large for the 32GB VRAM offload.\n"
                )
            self._is_loading = False
            self.eject_model()
            return False

        except Exception as e:
            import traceback

            error_msg = f"[server] Exception during model load: {e!s}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            with open(log_file, "a") as f:
                f.write(f"\nEXCEPTION: {error_msg}\n")
                traceback.print_exc(file=f)
            self._is_loading = False
            self.eject_model()
            return False

    def _kill_port_process(self) -> bool:
        import os
        import signal

        try:
            output = (
                subprocess
                .check_output(["lsof", "-t", f"-i:{settings.LLAMA_SERVER_PORT}"])
                .decode()
                .strip()
            )
            if output:
                killed = False
                for pid_str in output.split("\n"):
                    if pid_str.strip():
                        pid = int(pid_str.strip())
                        # Don't kill our own process
                        if pid == os.getpid():
                            continue
                        logger.info(
                            f"[server] Killing rogue process {pid} on port {settings.LLAMA_SERVER_PORT}"
                        )
                        try:
                            os.kill(pid, signal.SIGTERM)
                            # Wait a bit
                            time.sleep(1)
                            # Check if still running
                            os.kill(pid, 0)
                            # If no OSError was raised, it's still alive, so kill -9 it
                            os.kill(pid, signal.SIGKILL)
                        except OSError:
                            pass
                        killed = True
                return killed
        except Exception as e:
            logger.error(
                f"[server] Error killing process on port {settings.LLAMA_SERVER_PORT}: {e}"
            )
        return False

    def eject_model(self) -> bool:
        """Eject (unload) the currently loaded model."""
        if self._process is None and not self._port_in_use():
            logger.info("[server] No model currently loaded")
            self._current_model = None
            self._current_model_name = None
            self._current_params = {}
            self._is_loading = False
            return False

        logger.info("[server] Ejecting model...")
        self._is_loading = False
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("[server] Force killing...")
                self._process.kill()
                self._process.wait(timeout=2)
            self._process = None

        # Double check and clean up port just in case of rogue processes
        self._kill_port_process()
        self._current_model = None
        self._current_model_name = None
        self._current_params = {}
        logger.info("[server] Model ejected")
        return True

    def get_status(self) -> dict:
        status = {
            "running": self.is_running,
            "current_model": self._current_model,
            "current_model_name": self._current_model_name,
            "current_params": self._current_params,
            "is_loading": self._is_loading,
        }
        if self.is_running:
            try:
                import httpx

                resp = httpx.get(f"http://127.0.0.1:{settings.LLAMA_SERVER_PORT}/health", timeout=2)
                if resp.status_code == 200:
                    status["health"] = resp.json()
            except Exception as e:
                status["health_error"] = str(e)
        return status


server = ServerManager()
