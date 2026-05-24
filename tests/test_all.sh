#!/bin/bash
# LLamaStudio Integration Testing Suite Runner.
# This script sets the local environment flag and executes local GGUF tool-calling integration tests.

set -e

# Resolve script directory to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================================="
echo "🚀 Starting LlamaStudio GGUF Tool Calling Integration Tests"
echo "=========================================================="

export RUN_LOCAL_ONLY=1

# Run pytest inside project root
cd "$PROJECT_ROOT"
pytest -s -v tests/test_local_models.py

echo "=========================================================="
echo "🎉 GGUF Tool Calling Integration Tests Completed!"
echo "=========================================================="
