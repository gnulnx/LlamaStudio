# LLamaStudio

A desktop-like chat interface for llama.cpp, built with FastAPI + HTMX.

## Quick Start

```bash
cd /home/gnulnx/LLamaStuiod
python3 start.py
```

This will:
1. Start llama-server with your configured model
2. Start the FastAPI backend
3. Open a browser window automatically

## From Applications Menu

After installation, click "LLamaStudio" in your Pop!_OS Applications menu.

## Configuration

Edit `app/config.py` to change:
- Model path
- Server parameters (context size, GPU layers, etc.)
- Model directories to scan
- Default chat settings

## Architecture

- **FastAPI** backend with HTMX frontend (no JavaScript framework)
- **Server manager** handles llama-server lifecycle (start/stop/restart)
- **Chat manager** handles conversation state and streaming responses
- **Model discovery** scans GGUF directories for available models

## Project Structure

```
LLamaStuiod/
├── start.py              # Entry point - launches everything
├── app/
│   ├── __init__.py
│   ├── config.py         # Configuration
│   ├── main.py           # FastAPI app + endpoints
│   ├── server_manager.py # llama-server process manager
│   ├── chat.py           # Chat logic + llama.cpp API
│   └── templates/
│       └── index.html    # HTMX frontend
├── static/               # Static files
├── logs/                 # llama-server logs
├── llamastudio.desktop   # Pop!_OS launcher
└── llamastudio.svg       # App icon
```
