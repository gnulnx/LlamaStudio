"""Immutable tool permissions captured for one chat request."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from pathlib import Path

from .config_store import config_loader


@dataclass(frozen=True)
class SandboxPolicy:
    root: str
    disable_sandbox: bool
    command_policy: str

    @classmethod
    def capture(cls, root: str | Path | None = None) -> SandboxPolicy:
        settings = config_loader.get_workspace_config()
        return cls(
            str(Path(root or settings["root"]).expanduser().resolve()),
            bool(settings.get("disable_sandbox", False)),
            str(settings.get("command_policy", "allow")),
        )

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "disable_sandbox": self.disable_sandbox,
            "command_policy": self.command_policy,
        }


current_policy: contextvars.ContextVar[SandboxPolicy | None] = contextvars.ContextVar(
    "sandbox_policy", default=None
)


def effective_policy() -> SandboxPolicy:
    return current_policy.get() or SandboxPolicy.capture()
