# Testing LlamaStudio

Run the Python and frontend suites from the repository root:

```sh
python -m pytest -q --basetemp=.runtime/unit-tests
npm test -- --runInBand
```

The browser workflow job additionally runs Chromium against the real FastAPI
application, downloader, model process manager, chat pipeline and saved messages:

```sh
python -m playwright install chromium
RUN_BROWSER_TESTS=1 python -m pytest tests/test_browser_workflows.py -v --basetemp=.runtime/browser-tests
```

These integration tests use a local HTTP Hub and an external engine protocol
fixture. They verify actual transferred bytes, files, process arguments, image
payloads, rendered buttons, model loading from both web and TUI, failure recovery,
and API responsiveness. They do not measure GPU inference or model quality.
`PLAYWRIGHT_CHROMIUM_EXECUTABLE` can select an existing Chromium installation.

Unit tests exercise production code with fixtures at external boundaries. Avoid
copying implementation into a test or replacing the behavior being asserted with
a successful stub. The test audit replaced a copied parser, simulated downloads,
duplicated profile lookup logic, substring-only recovery checks, and CLI assertions
that incorrectly accepted errors as success. Parser tests now execute filesystem
tools and inspect persisted results; downloader tests use real HTTP transfers;
TUI profile tests mount the actual settings form. The browser tests reproduce the
quoted recovery-data failure and verify the overlay closes before image generation.

Native inference setup, backend restart, and package upgrade tests have separate
Linux/macOS CI jobs. Tests needing local GGUF models remain explicitly opt-in:

```sh
./tests/test_all.sh
```

The local matrix reads `local_test_models.json`, starts its own backend and CPU
engine with separate ports/configuration, and requires the model to execute an
actual `write_file` call. Missing configured models and failed CLI commands fail
the test. Both shell entrypoints propagate pytest failures and accept pytest
arguments. They do not eject the model in the user's running application.
