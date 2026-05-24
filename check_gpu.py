#!/usr/bin/env python3
import os
import sys

# Add the current directory to sys.path so we can import from the app package
sys.path.append(os.getcwd())

try:
    from app.gpu_utils import get_gpu_info

    info = get_gpu_info()
    print("-" * 30)
    print("GPU Detection Report")
    print("-" * 30)
    print(f"GPU Name: {info['name']}")
    print(f"VRAM:     {info['vram']} GB")
    print("-" * 30)
except Exception as e:
    print(f"Error detecting GPU: {e}")
    sys.exit(1)
