# LlamaStudio Developer & AI Agent Guidelines (AGENTS.md)

Welcome, AI Agent or developer! To maintain the architectural integrity, safety, and readability of the LlamaStudio repository, please adhere strictly to the following guidelines. 

---

## 🚀 Core Principles

1. **Prefer CLI Commands over One-Off Scripts**
   * If you need a script, utility, test harness, or configuration tool, **do not create loose Python/Bash scripts** in the repository root.
   * Instead, extend the unified `lls` command-line utility defined in [app/cli.py](file:///home/gnulnx/LlamaStudio/app/cli.py).
   * This maintains a single entry point for all administrative, testing, and operations tasks.

2. **Leverage the Rich CLI for Development and Testing**
   * Do not guess if the server is running or if a model is loaded. Always use:
     * `lls status` to view a styled dashboard of server, model parameters, and GPU memory state.
     * `lls eject` to gracefully unload a model and free VRAM before loading another.
     * `lls load <model_name>` to boot a specific model with customized parameters.
     * `lls oneshot "<prompt>"` to execute a real-time stream of thinking trace, content generation, and tool invocation.

3. **Workspace Paths Safety**
   * Any file operations (writing files, reading files, running shell commands) must strictly remain sandboxed within the workspace directory.
   * Always route path operations through the `check_path_safe(file_path)` validator implemented in [app/tools.py](file:///home/gnulnx/LlamaStudio/app/tools.py) to prevent directory traversal vulnerabilities.

---

## 🛠️ CLI Development Reference (`lls`)

The LlamaStudio CLI is built using `rich-click` for high-end visual styling.

### Installation
To link the entrypoint `lls` globally/locally in your environment:
```bash
pip install -e .
```

### Commands

| Command | Usage | Description |
| :--- | :--- | :--- |
| `status` | `lls status` | Visual dashboard of LlamaStudio API server, loaded model, and VRAM status. |
| `eject` | `lls eject` | Unload the active model to free GPU and CPU RAM. |
| `load` | `lls load [OPTIONS] MODEL` | Boot up a scanned model or direct GGUF file with overridden settings. |
| `oneshot` | `lls oneshot [OPTIONS] PROMPT` | Query the active model, streaming thinking/reasoning traces and local tool calls. |
| `reload` | `lls reload` | Gracefully restart/refresh the desktop application back-end. |

---

## 🧪 Integration Testing Suite

For verifying multi-model compatibility and tool calling robustness:
1. All local model integration tests are located in [tests/test_local_models.py](file:///home/gnulnx/LlamaStudio/tests/test_local_models.py).
2. The list of models to test is fully editable and defined in [tests/local_test_models.json](file:///home/gnulnx/LlamaStudio/tests/local_test_models.json).
3. These tests are **skipped by default** in typical environments and GitHub Actions (using `@pytest.mark.skipif`) to keep standard testing fast and self-contained.
4. To run the full integration suite locally:
   ```bash
   ./tests/test_all.sh
   ```

---

## 📐 Code Style and Standards
* **Python Style**: LlamaStudio uses Ruff for linting. Always run syntax checks before committing work.
* **Typing**: Use static type annotations (`from __future__ import annotations`) where appropriate.
* **Logging**: Direct logging calls through the custom logger module in `app/logger.py` rather than using raw `print` statements.
* **No Awol Coding**: Keep your changes minimal, modular, well-documented, and tightly scoped to requested features.
