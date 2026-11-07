"""Both local test entrypoints must propagate a failing test runner's exit code."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("script", ["test_all.sh", "test_all_bash.sh"])
def test_local_runner_propagates_failure_and_uses_the_shared_model_suite(tmp_path, script):
    runner = tmp_path / "pytest"
    recorded = tmp_path / "invocation.json"
    runner.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        f"open({str(recorded)!r},'w').write(json.dumps([os.environ.get('RUN_LOCAL_ONLY'),sys.argv[1:]]))\n"
        "sys.exit(7)\n"
    )
    runner.chmod(0o700)
    result = subprocess.run(
        ["bash", str(Path(__file__).parent / script), "-k", "chosen_model"],
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 7, result.stdout + result.stderr
    assert json.loads(recorded.read_text()) == [
        "1",
        [
            "-s",
            "-v",
            "tests/test_local_models.py",
            "--basetemp=.runtime/local-model-tests",
            "-k",
            "chosen_model",
        ],
    ]
