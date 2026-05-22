# LLamaStudio - Work Log

## Project Goal
Build a desktop-like chat interface for llama.cpp on Pop!_OS that feels seamless - with an Applications launcher, icon, and LM Studio-like model management.

## Hardware Constraint
**SINGLE GPU (RTX 5090, 32GB VRAM)** - only one model can be loaded at a time. Switching models requires ejecting the current one and loading a new one, which means downtime. Model switching is a deliberate user action, not automatic.

## Architecture
- FastAPI + HTMX backend
- llama.cpp server on port 1234
- App runs WITHOUT a model loaded - user explicitly loads models
- llama-server is only running when a model is loaded
- Model files in ~/.lmstudio/models/

## Current State

### Files
```
app/
  chat.py          - Chat logic, streaming, conversation management
  config.py        - Settings (server, chat, paths)
  main.py          - FastAPI app, API routes
  model_manager.py - Model discovery, scanning GGUF directories
  server_manager.py - llama-server lifecycle (load/eject models)
  templates/
    index.html     - Full HTMX frontend (chat + model browser)
start.py           - Startup script
llamastudio.desktop - Pop!_OS launcher
llamastudio.svg    - App icon
README.md          - Documentation
```

### Completed
- [x] Basic chat interface with streaming
- [x] Conversation management (new, switch, delete)
- [x] Model discovery (scans ~/.lmstudio/models/ for GGUF files)
- [x] Server manager with load_model() and eject_model() methods
- [x] Model browser in sidebar showing all available models
- [x] Pop!_OS launcher and icon installed
- [x] Settings panel (system prompt, temperature, top-p, max tokens)
- [x] Reasoning model support (reasoning_content)
- [x] Server status indicator in UI

### Current Issues
1. **main.py still calls server.start() on startup** - should NOT auto-start. App should launch clean with no model loaded.
2. **main.py has take_ownership() endpoint** - no longer needed with new architecture (app always owns the server)
3. **main.py has switch_model() endpoint** - should be replaced with load_model() + eject_model()
4. **Frontend still calls /api/models/switch/** - needs to call load/eject endpoints instead
5. **No "No model loaded" state** - frontend needs a clear visual state when no model is loaded
6. **No "Eject Model" button** - user needs a way to unload the current model
7. **Settings panel is incomplete** - missing many LM Studio-like settings

## Remaining Work - Tickets

### TICKET-1: App launches without a model loaded (CRITICAL)
**Priority:** Highest - blocks everything else
**What needs to happen:**
- Remove `server.start()` from startup event in main.py
- Remove `take_ownership()` endpoint and logic from main.py and server_manager.py
- Remove `switch_model()` endpoint from main.py
- Add `/api/models/load/{path}` POST endpoint that calls `server.load_model()`
- Add `/api/models/eject` POST endpoint that calls `server.eject_model()`
- Update startup event to just log "App started, no model loaded"
- Update shutdown event to call `server.eject_model()` if a model is loaded

### TICKET-2: Frontend "No model loaded" state (CRITICAL)
**Priority:** Highest - user must understand when no model is loaded
**What needs to happen:**
- Show "No model loaded" message in chat area when no model is loaded
- Disable send button when no model is loaded
- Show server status clearly (not just a tiny dot)
- Add "Eject Model" button in model browser when a model IS loaded
- Add "Load" button next to each model in the list
- When a model is loaded, show it prominently (like LM Studio's loaded model indicator)

### TICKET-3: More LM Studio-like settings (HIGH)
**Priority:** High - this is a core feature of the app
**What needs to happen:**
- Add settings for:
  - Context size (currently hardcoded)
  - GPU layers (currently hardcoded)
  - Flash attention (currently hardcoded)
  - KV cache type (currently hardcoded)
  - Stop sequences
  - Number of responses
  - Repetition penalty
- Settings should be editable in the UI and persist to config
- Settings changes should NOT require model reload (except GPU-related ones)

### TICKET-4: Model browser improvements (MEDIUM)
**Priority:** Medium - UX polish
**What needs to happen:**
- Show which model is currently loaded with a clear indicator (like LM Studio)
- Show model details on hover/click (size, quant, etc.)
- Add model search/filter in the list
- Add model tags/categories if possible
- Show loading progress when loading a model

### TICKET-5: Conversation persistence improvements (MEDIUM)
**Priority:** Medium - quality of life
**What needs to happen:**
- Save conversations persistently (currently in-memory)
- Restore conversations on app restart
- Show conversation count in sidebar
- Allow renaming conversations

### TICKET-6: App icon and integration polish (LOW)
**Priority:** Low - cosmetic
**What needs to happen:**
- Better SVG icon (current is basic)
- App window title bar integration
- System tray icon for running in background
- Keyboard shortcuts (Ctrl+N for new chat, etc.)

## Model Switching Flow (with single GPU)

```
User clicks "Load Model" on a model in the list:
  1. If no model loaded -> start llama-server with that model
  2. If a model IS loaded -> show confirmation dialog:
     "Eject [current_model] and load [new_model]?"
     (This is the only time we need the confirmation)
  3. Eject current model (kill llama-server process)
  4. Start llama-server with new model
  5. Show loading progress
  6. When ready, update UI to show new model loaded

User clicks "Eject Model":
  1. Kill llama-server process
  2. Show "No model loaded" state
  3. Disable chat input
```

## Known Issues
- Starlette 1.0.1 requires `request` as first arg to TemplateResponse
- Jinja2 absolute path for templates (relative paths cause 'unhashable type: dict' error)
- Reasoning models output `reasoning_content` not just `content`
- SSE parsing uses Python dict syntax, not JSON

## Dependencies
- Python 3.13
- FastAPI 0.136.1
- Starlette 1.0.1
- HTMX (via CDN)
- llama.cpp built from source (in /home/gnulnx/llama.cpp)
