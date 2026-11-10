#!/usr/bin/env bash
# Compatibility entrypoint: use the same assertions, model list and exit status.
set -euo pipefail
LLS_TEST_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$LLS_TEST_SCRIPT_DIR/test_all.sh" "$@"
