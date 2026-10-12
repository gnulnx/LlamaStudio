"""Guided inference setup before entering the terminal interface."""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import httpx
import rich_click as click
from rich.console import Console


def prepare_inference(
    api_url: str, console: Console, *, allow_restart: bool, restart: Callable[[], bool]
) -> None:
    """Use the connected backend's engine and installer, including remote backends."""
    if os.environ.get("CI"):
        return

    def status() -> httpx.Response:
        return httpx.get(f"{api_url}/api/inference/status", timeout=35)

    try:
        response = status()
        if response.status_code == 404:
            console.print(
                "The running backend predates inference setup; the package upgrade did not restart it.",
                markup=False,
            )
            current = httpx.get(f"{api_url}/api/server/status", timeout=3).json()
            if current.get("running") or current.get("is_loading") or not allow_restart:
                console.print(
                    "When the backend is idle, run lls reload to enable setup. Then run lls tui again.",
                    markup=False,
                )
                return
            if not click.confirm(
                "Restart the idle backend to enable local inference setup?", default=True
            ):
                return
            # Confirmation can take time; a model may have started meanwhile.
            current = httpx.get(f"{api_url}/api/server/status", timeout=3).json()
            if current.get("running") or current.get("is_loading"):
                console.print(
                    "A model session started; leaving the backend running. Reload when idle.",
                    markup=False,
                )
                return
            if not restart():
                raise click.ClickException("Could not restart the backend. Run lls reload.")
            # The launch helper can initially observe the old process before it exits.
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                try:
                    response = status()
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.25)
            else:
                raise click.ClickException("The backend did not refresh. Run lls reload and retry.")
        response.raise_for_status()
        state = response.json()
        if state.get("usable"):
            return
        if not state.get("running"):
            console.print(
                state.get("problem") or state.get("error") or "Local inference is not ready.",
                markup=False,
            )
            console.print(
                "Setup downloads a private engine. NVIDIA setup includes CUDA libraries (about 730 MiB).",
                markup=False,
            )
            if not click.confirm("Set up local inference now?", default=True):
                console.print("You can browse models now and run lls setup later.", markup=False)
                return
            response = httpx.post(
                f"{api_url}/api/inference/setup",
                headers={"X-LlamaStudio-Setup": "1"},
                timeout=10,
            )
            response.raise_for_status()
        deadline = time.monotonic() + 15 * 60
        with console.status("Setting up local inference...") as progress:
            while time.monotonic() < deadline:
                response = status()
                response.raise_for_status()
                state = response.json()
                progress.update(state.get("message") or "Setting up local inference...")
                if not state.get("running"):
                    if not state.get("usable"):
                        raise click.ClickException(
                            state.get("error")
                            or state.get("problem")
                            or "Inference setup failed. Run lls doctor."
                        )
                    console.print("Local inference is ready.", markup=False)
                    return
                time.sleep(0.5)
        raise click.ClickException(
            "Setup is still running. Check lls doctor before loading a model."
        )
    except (httpx.HTTPError, ValueError) as exc:
        raise click.ClickException(
            f"Could not check or set up local inference: {exc}. Run lls doctor."
        ) from exc
