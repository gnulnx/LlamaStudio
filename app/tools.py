"""
Safe Tool Executive for LLamaStudio.
Defines OpenAI-compatible schemas and implements execution for workspace-sandboxed tools.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .config_store import config_loader
from .logger import logger


def check_path_safe(file_path: str) -> Path:
    """Resolve file path and guarantee it remains strictly within the workspace root unless sandboxing is disabled."""
    target = Path(file_path)
    if config_loader.sandbox_disabled():
        return target.resolve()

    workspace_root = Path(config_loader.get_workspace_root()).resolve()
    # If relative, resolve against workspace root
    if not target.is_absolute():
        target = workspace_root / target

    target = target.resolve()

    # Check if target is indeed inside workspace_root
    if not str(target).startswith(str(workspace_root)):
        raise ValueError(
            f"Permission Denied: Target path '{file_path}' lies outside the workspace directory."
        )
    return target


def write_file(file_path: str, content: str) -> str:
    """Create or overwrite a file in the workspace directory with the specified content."""
    try:
        safe_path = check_path_safe(file_path)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)
        size = safe_path.stat().st_size
        return f"Successfully wrote {size} bytes to '{file_path}'"
    except Exception as e:
        logger.error(f"[tools] Error in write_file: {e}")
        return f"Error executing write_file: {e}"


def read_file(file_path: str) -> str:
    """Read and return the complete text content of a file in the workspace directory."""
    try:
        safe_path = check_path_safe(file_path)
        if not safe_path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not safe_path.is_file():
            return f"Error: '{file_path}' is a directory, not a file."
        with open(safe_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        return content
    except Exception as e:
        logger.error(f"[tools] Error in read_file: {e}")
        return f"Error executing read_file: {e}"


def list_dir(dir_path: str = ".") -> str:
    """List the names and types of files inside the specified workspace subdirectory."""
    try:
        safe_path = check_path_safe(dir_path)
        if not safe_path.exists():
            return f"Error: Directory '{dir_path}' does not exist."
        if not safe_path.is_dir():
            return f"Error: '{dir_path}' is a file, not a directory."

        workspace_root = Path(config_loader.get_workspace_root()).resolve()
        entries = []
        for p in safe_path.iterdir():
            try:
                rel = p.relative_to(workspace_root)
            except ValueError:
                rel = p.name
            suffix = "/" if p.is_dir() else ""
            size = f" ({p.stat().st_size} bytes)" if p.is_file() else ""
            entries.append(f"{rel}{suffix}{size}")

        if not entries:
            return f"Directory '{dir_path}' is empty."
        return "\n".join(sorted(entries))
    except Exception as e:
        logger.error(f"[tools] Error in list_dir: {e}")
        return f"Error executing list_dir: {e}"


def get_absolute_path(file_path: str = ".") -> str:
    """Return the absolute system path of a file or directory in the workspace directory."""
    try:
        safe_path = check_path_safe(file_path)
        return str(safe_path)
    except Exception as e:
        logger.error(f"[tools] Error in get_absolute_path: {e}")
        return f"Error executing get_absolute_path: {e}"


def run_command(command: str) -> str:
    """Execute a shell command inside the workspace root directory with a 15-second safety timeout."""
    try:
        logger.info(f"[tools] Executing command: {command}")
        workspace_root = Path(config_loader.get_workspace_root()).resolve()
        res = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace_root),
            capture_output=True,
            text=True,
            timeout=15.0,
        )
        output = []
        if res.stdout:
            output.append(f"--- Standard Output ---\n{res.stdout}")
        if res.stderr:
            output.append(f"--- Standard Error ---\n{res.stderr}")

        result_text = "\n".join(output) if output else "Command completed with no output."
        return f"Command returned exit code {res.returncode}\n{result_text}"
    except subprocess.TimeoutExpired:
        logger.warning(f"[tools] Command timed out: {command}")
        return "Error: Command timed out after 15 seconds."
    except Exception as e:
        logger.error(f"[tools] Error in run_command: {e}")
        return f"Error executing command: {e}"


# OpenAI compatible tool specifications
ALL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file in the workspace directory. Use this to create new files or completely overwrite existing files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to write, relative to the workspace directory (e.g. 'HelloFromMe.txt').",
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete content to write to the file.",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the complete contents of a text file inside the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to read, relative to the workspace directory.",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List all files and subdirectories inside a specific workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "The directory path to scan, relative to the workspace (defaults to '.').",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command locally in the workspace directory. Use this with care only when needed (e.g. running build scripts, checking git statuses).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute."}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_absolute_path",
            "description": "Return the absolute system path of a file or directory in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to resolve (relative to workspace). Defaults to '.' for workspace root.",
                    }
                },
            },
        },
    },
]


def execute_tool(name: str, arguments: dict) -> str:
    """Central tool dispatcher."""
    if name == "write_file":
        return write_file(arguments.get("file_path"), arguments.get("content"))
    elif name == "read_file":
        return read_file(arguments.get("file_path"))
    elif name == "list_dir":
        return list_dir(arguments.get("dir_path", "."))
    elif name == "run_command":
        return run_command(arguments.get("command"))
    elif name == "get_absolute_path":
        fp = arguments.get("file_path") or arguments.get("filename", ".")
        return get_absolute_path(fp)
    else:
        return f"Error: Tool '{name}' is not recognized."
