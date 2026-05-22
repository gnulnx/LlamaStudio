# 🦙 LLamaStudio

A beautiful, desktop-grade chat interface and local server manager for `llama.cpp`, crafted with **FastAPI** + **HTMX** for ultra-lightweight, zero-framework execution. Designed specifically for single-GPU setups (like Pop!_OS / Linux) to replicate the professional feel of LM Studio with seamless desktop integration.

---

## ✨ Features

- **⚡ Lightweight Frontend**: Powered by HTMX and Tailwind CSS (via CDN) with no bulky node modules or JS framework overhead.
- **🛠️ Single-GPU Optimized**: Designed around the hard constraint of single-GPU operations. Easily load, switch, and eject models.
- **📂 Model Discovery**: Scans local GGUF model directories (e.g. `~/.lmstudio/models`) automatically to present a clean model browser.
- **⚙️ Complete Settings Control**: Easily modify context size, GPU offloading layers, temperature, system prompts, flash attention, and KV cache quantization dynamically in the UI.
- **🪐 Process Management**: Automatically manages the lifecycle of the underlying `llama-server` process. Starts the server only when a model is loaded, and stops it cleanly on eject.
- **🖥️ Desktop Integration**: Includes a custom Pop!_OS / Ubuntu Applications launcher with a premium geometric vector icon.

---

## 🛠️ Prerequisites

1. **Python 3.13** (Recommended) or 3.10+
2. **Miniconda** or **Anaconda** (Highly recommended for package management)
3. **llama.cpp** built from source (or pre-compiled binaries):
   - By default, LLamaStudio expects the binary at `/home/gnulnx/llama.cpp/build/bin/llama-server`.
   - Update this path in `app/config.py` to match your local installation.

---

## 🚀 Installation & Setup

Follow these simple steps to set up and run LLamaStudio on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/LlamaStudio.git
cd LlamaStudio
```

### 2. Set Up the Conda Environment
Create and activate a clean conda environment:
```bash
conda create -n llamastudio python=3.13 -y
conda activate llamastudio
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Desktop Launcher (Optional)
To integrate the app seamlessly into your Linux desktop (Applications Menu):
```bash
# 1. Copy the desktop file to your local applications directory
cp llamastudio.desktop ~/.local/share/applications/

# 2. Copy the custom SVG icon to your local icons directory
mkdir -p ~/.local/share/icons/hicolor/128x128/apps/
cp llamastudio.svg ~/.local/share/icons/hicolor/128x128/apps/

# 3. Update your desktop database and icon cache
update-desktop-database ~/.local/share/applications/
gtk-update-icon-cache -f -t ~/.local/share/icons
```
*Note: The launcher is configured to run automatically using the conda environment Python at `/home/gnulnx/miniconda3/envs/llamastudio/bin/python`.*

---

## 💻 Running the App

### Via Desktop Menu (Pop!_OS / Ubuntu)
Simply search for **LLamaStudio** in your Applications menu, or press the Super key and type "Lla". Click the icon to start the server and open the interface automatically.

### Via Command Line
Activate your environment and run the startup script:
```bash
conda activate llamastudio
python start.py
```
This will start the FastAPI backend on `http://127.0.0.1:8765` and automatically launch your default web browser to the chat dashboard.

---

## ⚙️ Configuration

All major configurations can be found and edited in `app/config.py`:
- `LLAMA_SERVER_BIN`: The absolute path to your `llama-server` binary.
- `DEFAULT_MODEL`: The default GGUF model path to load if not specified.
- `MODEL_DIRS`: A list of directories to scan for GGUF model files.
- `APP_PORT`: The port the FastAPI web app runs on (defaults to `8765`).

```python
# app/config.py excerpt
LLAMA_SERVER_BIN: str = "/home/gnulnx/llama.cpp/build/bin/llama-server"
MODEL_DIRS: list[str] = [
    "/home/gnulnx/.lmstudio/models",
]
```

---

## 🏗️ Project Architecture

```
LlamaStudio/
├── start.py               # Main entrypoint (launches FastAPI + opens browser)
├── requirements.txt       # Declared python dependencies
├── llamastudio.desktop    # GNOME/Linux desktop launcher
├── llamastudio.svg        # Premium geometric custom application icon
├── app/
│   ├── config.py          # Settings & dynamic path configurations
│   ├── main.py            # FastAPI backend endpoints
│   ├── chat.py            # Chat streaming, prompt templates & API connectors
│   ├── model_manager.py   # Scans directories and parses GGUF files
│   ├── server_manager.py  # Handles llama-server process lifecycle
│   ├── logger.py          # Dedicated application logger
│   ├── tools.py           # Sandboxed workspace tool integrations
│   └── templates/
│       └── index.html     # Interactive HTMX frontend interface
```

---

## 📄 License

This project is open-source and available under the MIT License.
