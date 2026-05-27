# LlamaStudio Development Notes

## Runtime Config

User-facing runtime state is owned by `app.config_store.ConfigLoader`.

Default config directory:

```text
~/.config/llamastudio/
  config.json
  model_profiles.json
  conversations.json
  logs/
```

`app.config.Settings` remains the bootstrap/default layer. It resolves static defaults, ports, and environment overrides. Application code should prefer `ConfigLoader` for mutable runtime choices such as workspace permissions, model directories, launch state, llama defaults, chat defaults, and model profiles.

## Model Profiles

Per-model settings are first-class profile records in `model_profiles.json`, not UI-only blobs. Load paths from the web UI and CLI should resolve through `ConfigLoader` so a model tuned in the UI behaves the same when loaded with `lls load`.

The legacy `model_settings.json` format is migrated automatically when profiles are initialized.

## Launch Routing

`lls start` is the normal user entrypoint. It starts the app if needed and opens the browser to a state-aware view:

- first launch: Discover
- model already loaded: Chat
- local models available but no loaded model: Models
- no local models: Discover

The browser view is passed as `/?view=<name>` and the frontend applies it after initial state is loaded.

## Tests

Config behavior should be covered at the `ConfigLoader` boundary first. CLI and web tests should patch browser/server process actions rather than opening real browsers or starting real background servers.

## Local Hooks

Install the development hooks after `pip install -e ".[dev]"`:

```bash
pre-commit install
```

The hook runs Ruff lint fixes and formatting before commit, matching the GitHub Actions `ruff check .` and `ruff format --check .` gates.
