import contextlib
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import webbrowser
from pathlib import Path

import httpx
import rich_click as click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Configure rich-click visual styling to match a premium terminal theme
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.HEADER_COLOR = "cyan"
click.rich_click.OPTION_COLOR = "yellow"
click.rich_click.ARGUMENT_COLOR = "magenta"
click.rich_click.COMMAND_COLOR = "green"

console = Console()

# Resolve FastAPI server settings dynamically
try:
    from app.config import settings
    from app.config_store import config_loader
    from app.launch_context import app_url, local_connect_host, should_open_browser

    API_PORT = settings.APP_PORT
    API_HOST = settings.APP_HOST
except ImportError:
    API_PORT = 8765
    API_HOST = "127.0.0.1"
    app_url = None
    local_connect_host = None
    should_open_browser = None

API_CONNECT_HOST = local_connect_host(API_HOST) if local_connect_host else API_HOST
API_BASE_URL = f"http://{API_CONNECT_HOST}:{API_PORT}"


def server_launch_command() -> list[str]:
    """Return the importable server launcher command for source and wheel installs."""
    return [sys.executable, "-m", "app.launcher"]


def browser_url(view: str | None = None) -> str:
    if app_url is not None:
        return app_url(API_HOST, API_PORT, view)
    if view:
        return f"{API_BASE_URL}/?view={view}"
    return API_BASE_URL


def is_server_online() -> bool:
    """Check if the FastAPI app server is bound and listening on its designated port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        connect_host = local_connect_host(API_HOST) if local_connect_host else API_HOST
        return sock.connect_ex((connect_host, API_PORT)) == 0


def wait_for_server_ready(timeout: int = 15) -> bool:
    """Block and poll the API server health endpoint until it responds successfully."""
    start = time.time()
    url = f"{API_BASE_URL}/api/server/status"
    while time.time() - start < timeout:
        if is_server_online():
            try:
                resp = httpx.get(url, timeout=1.0)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
        time.sleep(0.5)
    return False


def start_server_background() -> bool:
    """Launch the main desktop FastAPI server as a daemonized background process."""
    console.print("[yellow]Starting LlamaStudio desktop server in the background...[/yellow]")
    subprocess.Popen(
        server_launch_command(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Wait for startup
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Waiting for API server to bind...", total=None)
        if wait_for_server_ready():
            progress.update(task, description="[green]API server is online and ready!")
            return True
        else:
            progress.update(task, description="[red]API server failed to start within timeout.")
            return False


def ensure_server_online() -> bool:
    """Ensure the desktop API is available for a CLI operation."""
    return is_server_online() or start_server_background()


def response_error(response: httpx.Response) -> str:
    """Extract a useful FastAPI error from an HTTP response."""
    try:
        payload = response.json()
        return str(payload.get("detail") or payload)
    except ValueError:
        return response.text or f"HTTP {response.status_code}"


def load_saved_model_settings(model_path: str) -> dict:
    """Load the per-model settings profile saved by the desktop UI."""
    return config_loader.get_model_profile_settings(model_path)


def select_launch_view_for_cli(consume_first_launch: bool = True) -> str:
    from app.model_manager import scan_models

    model_loaded = False
    if is_server_online():
        with contextlib.suppress(Exception):
            status_data = httpx.get(f"{API_BASE_URL}/api/server/status", timeout=2.0).json()
            model_loaded = bool(status_data.get("running"))

    return config_loader.get_launch_view(
        model_loaded=model_loaded,
        models_available=bool(scan_models()),
        consume_first_launch=consume_first_launch,
    )


@click.group()
def cli():
    """[cyan]LLamaStudio CLI (lls)[/cyan] - Manage your local LLMs and llama.cpp instances beautifully.

    Use this command-line utility to load/eject models, run real-time oneshot testing,
    and manage your background desktop server.
    """
    pass


@cli.command()
def start():
    """Start the desktop app and open the browser to the right launch view."""
    config_loader.initialize_for_launch(Path.cwd())

    if is_server_online():
        view = select_launch_view_for_cli(consume_first_launch=True)
        url = browser_url(view)
        if should_open_browser is None or should_open_browser():
            console.print(
                f"[green]LlamaStudio is already running.[/green] Opening [cyan]{url}[/cyan]"
            )
            webbrowser.open(url)
        else:
            console.print(f"[green]LlamaStudio is already running.[/green] Open [cyan]{url}[/cyan]")
        return

    if start_server_background():
        console.print(
            Panel(
                "[bold green]LLamaStudio started.[/bold green]\n\n"
                f"Web UI is available at [cyan]{browser_url()}[/cyan]\n"
                f"API server is bound on [cyan]http://{API_HOST}:{API_PORT}[/cyan]",
                border_style="green",
            )
        )
    else:
        console.print("[bold red]Failed to start LlamaStudio server.[/bold red]")


@cli.command()
def status():
    """Display the active server lifecycle state and loaded model metadata."""
    if not is_server_online():
        console.print(
            Panel(
                "[bold red]LlamaStudio Desktop Application / API Server is OFFLINE[/bold red]\n\n"
                f"Server is configured to run on [cyan]http://{API_HOST}:{API_PORT}[/cyan]\n"
                "Run [green]lls reload[/green], [green]lls load <model>[/green], "
                "or [green]llamastudio[/green] to launch it.",
                title="Status Dashboard",
                border_style="red",
            )
        )
        return

    try:
        # Query status and GPU endpoints
        status_data = httpx.get(f"{API_BASE_URL}/api/server/status").json()
        gpu_data = httpx.get(f"{API_BASE_URL}/api/gpu").json()

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_row("[bold cyan]Server URL[/bold cyan]", f"http://{API_HOST}:{API_PORT}")

        # Llama Server status
        running_state = status_data.get("running", False)
        running_lbl = (
            "[bold green]Active[/bold green]"
            if running_state
            else "[bold yellow]Idle[/bold yellow]"
        )
        table.add_row("[bold cyan]llama-server[/bold cyan]", running_lbl)

        # Loaded model info
        model_name = status_data.get("current_model_name")
        if model_name:
            table.add_row(
                "[bold cyan]Loaded Model[/bold cyan]", f"[bold blue]{model_name}[/bold blue]"
            )
            table.add_row("[bold cyan]Model Path[/bold cyan]", status_data.get("current_model", ""))
        else:
            table.add_row(
                "[bold cyan]Loaded Model[/bold cyan]",
                "[bold yellow]None (Ejected State)[/bold yellow]",
            )

        # Loading state
        if status_data.get("is_loading", False):
            table.add_row(
                "[bold cyan]Loading state[/bold cyan]",
                "[bold blink magenta]Booting new model...[/bold blink magenta]",
            )

        # Loaded params
        params = status_data.get("current_params", {})
        if params and running_state:
            param_details = ", ".join(f"{k}={v}" for k, v in params.items() if v is not None)
            table.add_row("[bold cyan]Model Params[/bold cyan]", f"[dim]{param_details}[/dim]")

        # GPU info
        gpu_name = gpu_data.get("name", "Unknown GPU")
        total_vram = gpu_data.get("total_vram", 0.0)
        free_vram = gpu_data.get("free_vram", 0.0)
        gpu_lbl = f"{gpu_name} ({free_vram:.2f} GB / {total_vram:.2f} GB free)"
        table.add_row("[bold cyan]Primary GPU[/bold cyan]", gpu_lbl)

        console.print(
            Panel(
                table,
                title="[bold green]LlamaStudio Status Dashboard[/bold green]",
                border_style="green",
            )
        )
    except Exception as e:
        console.print(f"[bold red]Failed to retrieve status: {e}[/bold red]")


@cli.command()
def eject():
    """Unload the active model and completely free GPU/CPU memory."""
    if not is_server_online():
        console.print(
            "[bold yellow]LlamaStudio server is offline. No model is currently loaded.[/bold yellow]"
        )
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ejecting model...", total=None)
        try:
            resp = httpx.post(f"{API_BASE_URL}/api/models/eject", timeout=15.0)
            if resp.status_code == 200:
                progress.update(task, description="[green]Model successfully ejected!")
                console.print(
                    Panel(
                        "[bold green]VRAM and System RAM have been successfully freed.[/bold green]\n"
                        "LlamaStudio backend has entered an Idle state.",
                        border_style="green",
                    )
                )
            else:
                progress.update(task, description="[red]Ejection failed.")
                console.print(f"[bold red]Server returned error: {resp.text}[/bold red]")
        except Exception as e:
            progress.update(task, description="[red]Connection error.")
            console.print(f"[bold red]Failed to communicate with API server: {e}[/bold red]")


@cli.command(name="ls")
def list_models_cmd():
    """List all available models in your scanned directories."""
    from app.model_manager import scan_models

    scanned_models = scan_models()
    if not scanned_models:
        console.print("[bold yellow]No models found in your scanned directories.[/bold yellow]")
        return

    table = Table(title="[bold green]Available Models[/bold green]", border_style="cyan")
    table.add_column("No.", justify="right", style="yellow")
    table.add_column("Model Name", style="bold cyan")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Quantization", style="magenta")

    for idx, m in enumerate(scanned_models, start=1):
        table.add_row(
            str(idx),
            m.name,
            m.size_human,
            m.quant or "Unknown",
        )

    console.print(table)


@cli.command()
@click.argument("model", required=False)
@click.option("--ctx-size", type=int, help="Override context size")
@click.option("--gpu-layers", type=int, help="Override offloaded GPU layers")
@click.option("--threads", type=int, help="Number of CPU threads to use")
@click.option(
    "--chat-template",
    type=click.Choice(["chatml", "gemma", "llama3", "deepseek-r1", "custom"]),
    help="Override the prompt chat template schema",
)
@click.option("--custom-template", help="Custom Jinja template definition string")
@click.option(
    "--flash-attn", type=click.Choice(["on", "off"]), help="Enable or disable Flash Attention"
)
@click.option("--kv-cache-type", help="Quantization type for Key-Value cache (e.g. q8_0, f16)")
@click.option("--vocab-type", help="Quantization type for vocabulary (e.g. q8_0, f16)")
@click.option("--override-kv", help="Format: key=type:val override string")
@click.option("--task-timeout", type=int, help="Override llama-server task timeout in seconds")
@click.option("--cpu-mode", is_flag=True, help="Force CPU inference (sets gpu-layers=0)")
@click.option(
    "--no-saved-settings",
    is_flag=True,
    help="Ignore saved desktop UI settings for this model",
)
@click.option(
    "--reload", is_flag=True, help="Restart/Reload the desktop application before loading"
)
def load(model, reload, **kwargs):
    """Load a specific model with customized parameters.

    MODEL can be a scanned model name (e.g. 'gemma-4-26B-A4B-it-Q8_0') or a full file path to a GGUF file.
    If MODEL is omitted, LlamaStudio displays a numbered list of available models to select from.
    """
    # 1. Resolve model name/path
    from app.model_manager import scan_models

    scanned_models = scan_models()

    resolved_path = None
    model_name = None

    if not model:
        if not scanned_models:
            console.print("[bold red]Error: No models found in scanned directories.[/bold red]")
            return

        console.print("[bold cyan]Available Scanned Models:[/bold cyan]")
        for idx, m in enumerate(scanned_models, start=1):
            console.print(
                f"  [bold yellow]{idx}[/bold yellow]. {m.name} [dim]({m.size_human})[/dim]"
            )

        selection = click.prompt("\nSelect a model number to load", type=int)
        if selection < 1 or selection > len(scanned_models):
            console.print("[bold red]Invalid selection.[/bold red]")
            return

        selected_model = scanned_models[selection - 1]
        resolved_path = selected_model.path
        model_name = selected_model.name
    else:
        # Try direct path
        if Path(model).exists() and Path(model).is_file():
            resolved_path = str(Path(model).resolve())
            model_name = Path(model).stem
        else:
            # Search by scanned name
            for m in scanned_models:
                if m.name == model or m.path == model:
                    resolved_path = m.path
                    model_name = m.name
                    break

            # Fuzzy search case-insensitive contains
            if not resolved_path:
                for m in scanned_models:
                    if model.lower() in m.name.lower():
                        resolved_path = m.path
                        model_name = m.name
                        console.print(
                            f"[yellow]Fuzzy matched model to: [bold cyan]{model_name}[/bold cyan][/yellow]"
                        )
                        break

        if not resolved_path:
            console.print(f"[bold red]Error: Could not find or resolve model '{model}'[/bold red]")
            if scanned_models:
                console.print("\n[bold cyan]Available Scanned Models:[/bold cyan]")
                for m in scanned_models:
                    console.print(f"  - {m.name} [dim]({m.size_human})[/dim]")
            return

    # 2. Handle reload or server offline
    if reload and is_server_online():
        console.print("[yellow]Reload option specified. Restarting LlamaStudio server...[/yellow]")
        subprocess.Popen(
            server_launch_command(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(1)  # let port close
        if not wait_for_server_ready(20):
            console.print("[bold red]Failed to reload and start FastAPI server.[/bold red]")
            return
    elif not is_server_online():
        if not start_server_background():
            return

    # 3. Compile custom settings parameters. Start with the UI profile so CLI loads
    # do not discard tuned context/KV/cache settings, then apply explicit overrides.
    settings_payload = (
        {} if kwargs.get("no_saved_settings") else load_saved_model_settings(resolved_path)
    )
    if settings_payload:
        console.print("[dim]Using saved load settings profile for this model.[/dim]")

    if kwargs.get("ctx_size") is not None:
        settings_payload["ctx_size"] = kwargs["ctx_size"]
    if kwargs.get("gpu_layers") is not None:
        settings_payload["gpu_layers"] = kwargs["gpu_layers"]
    if kwargs.get("threads") is not None:
        settings_payload["threads"] = kwargs["threads"]
    if kwargs.get("chat_template") is not None:
        settings_payload["chat_template"] = kwargs["chat_template"]
    if kwargs.get("custom_template") is not None:
        settings_payload["custom_template"] = kwargs["custom_template"]
    if kwargs.get("flash_attn") is not None:
        settings_payload["flash_attn"] = kwargs["flash_attn"]
    if kwargs.get("kv_cache_type") is not None:
        settings_payload["kv_cache_type"] = kwargs["kv_cache_type"]
    if kwargs.get("vocab_type") is not None:
        settings_payload["vocab_type"] = kwargs["vocab_type"]
    if kwargs.get("override_kv") is not None:
        settings_payload["override_kv"] = kwargs["override_kv"]
    if kwargs.get("task_timeout") is not None:
        settings_payload["task_timeout"] = kwargs["task_timeout"]
    if kwargs.get("cpu_mode"):
        settings_payload["cpu_mode"] = True
        settings_payload["gpu_layers"] = 0

    # 4. Trigger model load API
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Loading model '{model_name}' (this can take up to 2 minutes)...", total=None
        )
        try:
            payload = {"path": resolved_path, "settings": settings_payload}
            # Timeout is extended to 180s for massive GGUF loading
            resp = httpx.post(f"{API_BASE_URL}/api/models/load", json=payload, timeout=180.0)
            if resp.status_code == 200:
                progress.update(task, description="[green]Model successfully loaded!")
                console.print(
                    Panel(
                        f"[bold green]LlamaServer is running! Model loaded:[/bold green]\n"
                        f"[bold blue]{model_name}[/bold blue]\n\n"
                        f"FastAPI Server is online at [cyan]http://{API_HOST}:{API_PORT}[/cyan]",
                        border_style="green",
                    )
                )
            else:
                progress.update(task, description="[red]Loading failed.")
                console.print(f"[bold red]Server returned error: {resp.text}[/bold red]")
        except Exception as e:
            progress.update(task, description="[red]Connection error.")
            console.print(f"[bold red]Failed to load model: {e}[/bold red]")


@cli.command()
@click.argument("prompt")
@click.option("-m", "--model", help="Optional GGUF model name or path to load before execution")
@click.option("--system-prompt", help="System prompt rules override")
@click.option("--temperature", type=float, help="Override inference temperature")
@click.option("--top-p", type=float, help="Override Nucleus Sampling top_p")
@click.option("--max-tokens", type=int, help="Override maximum tokens output constraint")
@click.option(
    "--thinking/--no-thinking",
    default=None,
    help="Enable or disable model reasoning for this request without reloading the model",
)
@click.option(
    "--image",
    "image_paths",
    multiple=True,
    help="Workspace image path to attach; may be provided more than once",
)
@click.option(
    "--audio",
    "audio_path",
    help="Workspace WAV, MP3, or FLAC path to attach as model audio input",
)
def oneshot(prompt, model, **kwargs):
    """Execute a single testing query against a model, showing real-time reasoning and tool outputs."""
    # 1. Load model if specified
    if model:
        # Call load command logic programmatically
        from click.testing import CliRunner

        runner = CliRunner()
        console.print(f"[cyan]Ensuring model '{model}' is loaded...[/cyan]")
        resp = runner.invoke(load, [model])
        if resp.exit_code != 0:
            console.print(
                f"[bold red]Failed to auto-load model '{model}': {resp.output}[/bold red]"
            )
            return

    # 2. Verify server is online and has a running model
    if not is_server_online():
        console.print(
            "[bold red]Error: LlamaStudio API server is offline. Load a model first using 'lls load <model>'.[/bold red]"
        )
        return

    try:
        status_data = httpx.get(f"{API_BASE_URL}/api/server/status").json()
        if not status_data.get("running"):
            console.print(
                "[bold red]Error: No model is currently loaded in the server. Run 'lls load <model>' first.[/bold red]"
            )
            return
    except Exception as e:
        console.print(f"[bold red]Failed to contact LlamaStudio server: {e}[/bold red]")
        return

    image_payloads = []
    if kwargs.get("image_paths"):
        from app.tools import ToolResult, read_file

        for image_path in kwargs["image_paths"]:
            result = read_file(image_path)
            if not isinstance(result, ToolResult) or not result.images:
                detail = result if isinstance(result, str) else "Unsupported image."
                console.print(f"[bold red]Could not attach '{image_path}': {detail}[/bold red]")
                return
            image_payloads.extend(result.images)

    audio_payloads = []
    if kwargs.get("audio_path"):
        from app.tools import ToolResult, read_file

        audio_path = kwargs["audio_path"]
        result = read_file(audio_path)
        if not isinstance(result, ToolResult) or not result.audios:
            detail = result if isinstance(result, str) else "Unsupported audio."
            console.print(f"[bold red]Could not attach '{audio_path}': {detail}[/bold red]")
            return
        audio_payloads.extend(result.audios)

    # Start a fresh conversation to avoid history pollution across sequential oneshot runs
    with contextlib.suppress(Exception):
        httpx.post(f"{API_BASE_URL}/api/chat/new")

    # 3. Construct chat payload
    payload = {"message": prompt}
    if image_payloads:
        payload["images"] = image_payloads
    if audio_payloads:
        payload["audios"] = audio_payloads
    if kwargs.get("system_prompt") is not None:
        payload["system_prompt"] = kwargs["system_prompt"]
    if kwargs.get("temperature") is not None:
        payload["temperature"] = kwargs["temperature"]
    if kwargs.get("top_p") is not None:
        payload["top_p"] = kwargs["top_p"]
    if kwargs.get("max_tokens") is not None:
        payload["max_tokens"] = kwargs["max_tokens"]
    if kwargs.get("thinking") is not None:
        payload["enable_thinking"] = kwargs["thinking"]

    # 4. Stream chat completions using httpx
    console.print(
        "\n[bold cyan]─── LlamaStudio Chat Stream ──────────────────────────[/bold cyan]\n"
    )
    try:
        with httpx.stream(
            "POST", f"{API_BASE_URL}/api/chat/send", json=payload, timeout=300.0
        ) as r:
            if r.status_code != 200:
                console.print(f"[bold red]Error from server: {r.status_code}[/bold red]")
                with contextlib.suppress(Exception):
                    console.print(r.read().decode())
                return

            in_reasoning = False
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]  # Strip "data: "

                # Check for stream completion
                if data_str.strip() == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)

                    # Handle error event
                    if data.get("error"):
                        console.print(f"\n[bold red]Server Error: {data['error']}[/bold red]")
                        break

                    if data.get("type") == "start":
                        continue

                    if data.get("type") == "metrics":
                        metrics = data.get("metrics") or {}
                        total = metrics.get("total_tokens")
                        prompt_tokens = metrics.get("prompt_tokens")
                        completion_tokens = metrics.get("completion_tokens")
                        elapsed = metrics.get("elapsed_seconds")
                        tokens_per_second = metrics.get("tokens_per_second")
                        parts = []
                        if total is not None:
                            parts.append(f"{total} tokens")
                        if prompt_tokens is not None and completion_tokens is not None:
                            parts.append(f"{prompt_tokens} in / {completion_tokens} out")
                        if elapsed is not None:
                            parts.append(f"{elapsed:.2f}s")
                        if tokens_per_second is not None:
                            parts.append(f"{tokens_per_second:.1f} tok/s")
                        if parts:
                            if in_reasoning:
                                in_reasoning = False
                                console.print("\n")
                            console.print(f"\n[dim]{' · '.join(parts)}[/dim]")
                        continue

                    if data.get("type") == "vision_error":
                        console.print(f"\n[bold red]Vision Error: {data.get('message')}[/bold red]")
                        recovery = data.get("recovery") or {}
                        if recovery.get("status") == "downloadable":
                            console.print(
                                f"[yellow]Matching projector: {recovery.get('filename')}[/yellow]"
                            )
                        continue

                    if data.get("type") == "audio_error":
                        console.print(f"\n[bold red]Audio Error: {data.get('message')}[/bold red]")
                        continue

                    # Handle DeepSeek Chain-of-Thought reasoning
                    reasoning = data.get("reasoning")
                    if reasoning:
                        if not in_reasoning:
                            in_reasoning = True
                            console.print("[bold yellow]🧠 Thought trace:[/bold yellow]")
                        console.print(reasoning, end="", style="dim yellow")
                        continue

                    # Handle standard text content
                    content = data.get("content")
                    if content:
                        if in_reasoning:
                            in_reasoning = False
                            console.print("\n")  # newline separator
                        console.print(content, end="")
                        continue

                    # Handle local safe tool calls triggered on the server
                    if data.get("type") == "tool_exec_start":
                        tool_name = data.get("name")
                        tool_args = data.get("arguments")
                        if in_reasoning:
                            in_reasoning = False
                            console.print("\n")
                        console.print(
                            f"\n[bold blue]🔧 Executing tool: [magenta]{tool_name}[/magenta] with args: {json.dumps(tool_args)}[/bold blue]"
                        )
                        continue

                    if data.get("type") == "tool_exec_end":
                        tool_name = data.get("name")
                        tool_result = data.get("result")
                        console.print(
                            f"[bold green]✅ Tool '{tool_name}' completed. Result:[/bold green]\n[dim]{tool_result}[/dim]"
                        )
                        continue

                    if data.get("type") == "end":
                        break
                except Exception:
                    continue
        console.print(
            "\n\n[bold cyan]──────────────────────────────────────────────────────[/bold cyan]\n"
        )
    except Exception as e:
        console.print(f"\n[bold red]Network/Inference failure: {e}[/bold red]")


@cli.group()
def speech():
    """Install, manage, and use local Whisper speech-to-text."""


@speech.command(name="status")
def speech_status():
    """Show Whisper installation, model, and server state."""
    try:
        if is_server_online():
            response = httpx.get(f"{API_BASE_URL}/api/speech/status", timeout=5.0)
            response.raise_for_status()
            status_data = response.json()
        else:
            from app.speech_manager import speech as speech_engine

            status_data = speech_engine.get_status()

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_row("[bold cyan]Engine[/bold cyan]", status_data["engine"])
        table.add_row("[bold cyan]Version[/bold cyan]", status_data["version"])
        table.add_row(
            "[bold cyan]Installation[/bold cyan]",
            "[green]Ready[/green]"
            if status_data["installed"]
            else "[yellow]Not installed[/yellow]",
        )
        table.add_row("[bold cyan]Model[/bold cyan]", status_data["model"])
        table.add_row(
            "[bold cyan]Model file[/bold cyan]",
            (
                "[green]Ready[/green]"
                if status_data["model_installed"]
                else "[yellow]Not installed[/yellow]"
            ),
        )
        table.add_row(
            "[bold cyan]Speech server[/bold cyan]",
            "[green]Running[/green]" if status_data["running"] else "[yellow]Stopped[/yellow]",
        )
        table.add_row("[bold cyan]Compute[/bold cyan]", "GPU" if status_data["use_gpu"] else "CPU")
        if status_data.get("binary"):
            table.add_row("[bold cyan]Binary[/bold cyan]", status_data["binary"])
        table.add_row("[bold cyan]Model path[/bold cyan]", status_data["model_path"])
        console.print(Panel(table, title="[bold green]LlamaStudio Speech[/bold green]"))
        if not status_data["installed"] or not status_data["model_installed"]:
            console.print(
                f"Run [green]lls speech install --model {status_data['model']}[/green] to set it up."
            )
    except Exception as exc:
        console.print(f"[bold red]Could not retrieve speech status: {exc}[/bold red]")


@speech.command(name="install")
@click.option(
    "--model",
    type=click.Choice(["base.en", "small.en", "large-v3-turbo-q5_0"]),
    default="small.en",
    show_default=True,
    help="Whisper model to download.",
)
@click.option(
    "--install-dir",
    type=click.Path(path_type=Path),
    help="Override the managed speech runtime directory.",
)
def speech_install(model: str, install_dir: Path | None):
    """Install a pinned whisper.cpp binary and checksum-verified model."""
    from app.speech_manager import speech as speech_engine

    tasks: dict[str, int] = {}
    try:
        with Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            console=console,
        ) as progress:

            def update_download(done: int, total: int | None, label: str) -> None:
                if label not in tasks:
                    tasks[label] = progress.add_task(label, total=total)
                progress.update(tasks[label], completed=done, total=total)

            result = speech_engine.install(
                model=model,
                install_dir=install_dir,
                progress=update_download,
            )
        console.print(
            Panel(
                "[bold green]Local speech-to-text is ready.[/bold green]\n"
                f"Model: [cyan]{result['model']}[/cyan]\n"
                f"Runtime: [dim]{result['install_dir']}[/dim]\n\n"
                "Try [green]lls speech record[/green] or use the microphone in chat.",
                border_style="green",
            )
        )
    except Exception as exc:
        console.print(f"[bold red]Speech installation failed: {exc}[/bold red]")
        raise click.exceptions.Exit(1) from exc


@speech.command(name="load")
@click.argument("model", required=False)
@click.option(
    "--gpu/--cpu",
    default=None,
    help="Run Whisper on the GPU or CPU. The default uses the saved setting.",
)
def speech_load(model: str | None, gpu: bool | None):
    """Start the persistent Whisper transcription server."""
    if not ensure_server_online():
        raise click.exceptions.Exit(1)
    try:
        response = httpx.post(
            f"{API_BASE_URL}/api/speech/load",
            json={"model": model, "use_gpu": gpu},
            timeout=90.0,
        )
        if response.status_code != 200:
            raise RuntimeError(response_error(response))
        data = response.json()
        console.print(
            f"[bold green]Whisper is ready.[/bold green] "
            f"Model [cyan]{data['model']}[/cyan] on "
            f"[cyan]{'GPU' if data['use_gpu'] else 'CPU'}[/cyan]."
        )
    except Exception as exc:
        console.print(f"[bold red]Could not start speech engine: {exc}[/bold red]")
        raise click.exceptions.Exit(1) from exc


@speech.command(name="eject")
def speech_eject():
    """Unload Whisper and release its memory."""
    if not is_server_online():
        console.print(
            "[yellow]LlamaStudio is offline; no managed speech server is active.[/yellow]"
        )
        return
    try:
        response = httpx.post(f"{API_BASE_URL}/api/speech/eject", timeout=10.0)
        if response.status_code != 200:
            raise RuntimeError(response_error(response))
        console.print("[bold green]Whisper has been unloaded.[/bold green]")
    except Exception as exc:
        console.print(f"[bold red]Could not stop speech engine: {exc}[/bold red]")
        raise click.exceptions.Exit(1) from exc


def request_transcription(
    audio: bytes,
    *,
    content_type: str,
    language: str,
    translate: bool,
) -> str:
    """Send encoded audio to the LlamaStudio speech API and return text."""
    if not ensure_server_online():
        raise RuntimeError("LlamaStudio API server could not be started.")
    response = httpx.post(
        f"{API_BASE_URL}/api/speech/transcribe",
        params={"language": language, "translate": str(translate).lower()},
        content=audio,
        headers={"Content-Type": content_type},
        timeout=300.0,
    )
    if response.status_code != 200:
        raise RuntimeError(response_error(response))
    return str(response.json()["text"]).strip()


@speech.command(name="transcribe")
@click.argument("audio_path", type=click.Path(path_type=Path))
@click.option("--language", default="auto", show_default=True)
@click.option("--translate", is_flag=True, help="Translate recognized speech to English.")
def speech_transcribe(audio_path: Path, language: str, translate: bool):
    """Transcribe an audio file inside the configured workspace."""
    from app.tools import check_path_safe

    try:
        safe_path = check_path_safe(str(audio_path))
        if not safe_path.is_file():
            raise RuntimeError(f"Audio file not found: {audio_path}")
        suffix_types = {
            ".flac": "audio/flac",
            ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg",
            ".wav": "audio/wav",
            ".webm": "audio/webm",
            ".m4a": "audio/mp4",
        }
        text = request_transcription(
            safe_path.read_bytes(),
            content_type=suffix_types.get(safe_path.suffix.lower(), "application/octet-stream"),
            language=language,
            translate=translate,
        )
        console.print(text)
    except Exception as exc:
        console.print(f"[bold red]Transcription failed: {exc}[/bold red]")
        raise click.exceptions.Exit(1) from exc


@speech.command(name="record")
@click.option("--device", default="default", show_default=True, help="ALSA capture device.")
@click.option("--language", default="auto", show_default=True)
@click.option("--translate", is_flag=True, help="Translate recognized speech to English.")
def speech_record(device: str, language: str, translate: bool):
    """Record now, press Enter to stop, then print the transcript."""
    from app.tools import check_path_safe

    recorder = shutil.which("arecord")
    if recorder is None:
        console.print("[bold red]arecord is required for terminal microphone capture.[/bold red]")
        raise click.exceptions.Exit(1)

    workspace = check_path_safe(".")
    with tempfile.NamedTemporaryFile(
        prefix=".lls-speech-",
        suffix=".wav",
        dir=workspace,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    process: subprocess.Popen | None = None
    try:
        process = subprocess.Popen(
            [
                recorder,
                "--quiet",
                "--device",
                device,
                "--format",
                "S16_LE",
                "--rate",
                "16000",
                "--channels",
                "1",
                "--file-type",
                "wav",
                str(temporary_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        console.print(
            "[bold red]● Recording[/bold red] — press [bold]Enter[/bold] to stop and transcribe."
        )
        click.getchar()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

        if temporary_path.stat().st_size <= 44:
            detail = (process.stderr.read() if process.stderr else b"").decode(
                "utf-8", errors="replace"
            )
            raise RuntimeError(detail.strip() or "The microphone recording was empty.")
        text = request_transcription(
            temporary_path.read_bytes(),
            content_type="audio/wav",
            language=language,
            translate=translate,
        )
        console.print(Panel(text, title="[bold green]Transcript[/bold green]"))
    except KeyboardInterrupt:
        console.print("\n[yellow]Recording cancelled.[/yellow]")
    except Exception as exc:
        console.print(f"[bold red]Recording failed: {exc}[/bold red]")
        raise click.exceptions.Exit(1) from exc
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=2)
        temporary_path.unlink(missing_ok=True)


@cli.command()
def reload():
    """Force stop and restart the Desktop Application back-end."""
    config_loader.initialize_for_launch(Path.cwd())
    if is_server_online():
        console.print(
            "[yellow]Active LlamaStudio server detected. Triggering graceful reload...[/yellow]"
        )
    else:
        console.print("[yellow]Server is offline. Starting fresh application...[/yellow]")

    if start_server_background():
        browser_message = (
            "A new web browser tab has been launched automatically."
            if should_open_browser is None or should_open_browser()
            else f"Browser launch skipped. Open [cyan]{browser_url()}[/cyan] manually."
        )
        console.print(
            Panel(
                "[bold green]LLamaStudio successfully reloaded![/bold green]\n\n"
                f"Web UI is available at [cyan]{browser_url()}[/cyan]\n"
                f"API server is bound on [cyan]http://{API_HOST}:{API_PORT}[/cyan]\n"
                f"{browser_message}",
                border_style="green",
            )
        )
    else:
        console.print("[bold red]Failed to reload LlamaStudio server.[/bold red]")


if __name__ == "__main__":
    cli()
