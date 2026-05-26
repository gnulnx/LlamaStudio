import contextlib
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import rich_click as click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
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

    API_PORT = settings.APP_PORT
    API_HOST = settings.APP_HOST
except ImportError:
    API_PORT = 8765
    API_HOST = "127.0.0.1"

API_BASE_URL = f"http://{API_HOST}:{API_PORT}"


def server_launch_command() -> list[str]:
    """Return the importable server launcher command for source and wheel installs."""
    return [sys.executable, "-m", "app.launcher"]


def is_server_online() -> bool:
    """Check if the FastAPI app server is bound and listening on its designated port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((API_HOST, API_PORT)) == 0


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


@click.group()
def cli():
    """[cyan]LLamaStudio CLI (lls)[/cyan] - Manage your local LLMs and llama.cpp instances beautifully.

    Use this command-line utility to load/eject models, run real-time oneshot testing,
    and manage your background desktop server.
    """
    pass


@cli.command()
def status():
    """Display the active server lifecycle state and loaded model metadata."""
    if not is_server_online():
        console.print(
            Panel(
                "[bold red]LlamaStudio Desktop Application / API Server is OFFLINE[/bold red]\n\n"
                f"Server is configured to run on [cyan]http://{API_HOST}:{API_PORT}[/cyan]\n"
                "Run [green]lls load <model>[/green] or [green]python3 start.py[/green] to launch it.",
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
@click.option("--ctx-size", type=int, help="Override context size (default: 16384)")
@click.option("--gpu-layers", type=int, help="Override offloaded GPU layers (default: 999)")
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
@click.option("--cpu-mode", is_flag=True, help="Force CPU inference (sets gpu-layers=0)")
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

    # 3. Compile custom settings parameters
    settings_payload = {}
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

    # Start a fresh conversation to avoid history pollution across sequential oneshot runs
    with contextlib.suppress(Exception):
        httpx.post(f"{API_BASE_URL}/api/chat/new")

    # 3. Construct chat payload
    payload = {"message": prompt}
    if kwargs.get("system_prompt") is not None:
        payload["system_prompt"] = kwargs["system_prompt"]
    if kwargs.get("temperature") is not None:
        payload["temperature"] = kwargs["temperature"]
    if kwargs.get("top_p") is not None:
        payload["top_p"] = kwargs["top_p"]
    if kwargs.get("max_tokens") is not None:
        payload["max_tokens"] = kwargs["max_tokens"]

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


@cli.command()
def reload():
    """Force stop and restart the Desktop Application back-end."""
    if is_server_online():
        console.print(
            "[yellow]Active LlamaStudio server detected. Triggering graceful reload...[/yellow]"
        )
    else:
        console.print("[yellow]Server is offline. Starting fresh application...[/yellow]")

    if start_server_background():
        console.print(
            Panel(
                "[bold green]LLamaStudio successfully reloaded![/bold green]\n\n"
                f"Web UI and API server is live on [cyan]http://{API_HOST}:{API_PORT}[/cyan]\n"
                "A new web browser tab has been launched automatically.",
                border_style="green",
            )
        )
    else:
        console.print("[bold red]Failed to reload LlamaStudio server.[/bold red]")


if __name__ == "__main__":
    cli()
