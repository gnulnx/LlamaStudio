// Polyfill TextEncoder and TextDecoder for JSDOM in Jest environment
const { TextEncoder, TextDecoder } = require('util');
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

describe("Frontend Chat Settings and Stream Stop Features", () => {
    let scriptCode;
    let dom;
    let window;

    beforeAll(() => {
        // Load index.html
        const htmlPath = path.resolve(__dirname, '../../app/templates/index.html');
        const htmlContent = fs.readFileSync(htmlPath, 'utf8');

        // Extract the content of the <script> block
        const scriptMatch = htmlContent.match(/<script>([\s\S]*?)<\/script>/);
        if (!scriptMatch) {
            throw new Error("Could not find <script> block in index.html");
        }

        let rawScript = scriptMatch[1];

        // Replace Jinja2 template tags with mock JavaScript values
        rawScript = rawScript
            .replace(/let conversations = \{\{ conversations \| tojson \}\};/, 'let conversations = [];')
            .replace(/let loadedModelPath = '\{\{ current_model or "" \}\}';/, 'let loadedModelPath = "";')
            .replace(/let loadedModelName = '\{\{ current_model_name or "" \}\}';/, 'let loadedModelName = "";')
            .replace(/let isServerRunning = \{% if server_running %\}true\{% else %\}false\{% endif %\};/, 'let isServerRunning = false;');

        scriptCode = rawScript;
    });

    beforeEach(() => {
        // Setup JSDOM with necessary elements for chat features
        dom = new JSDOM(`
            <button id="toggleChatParamsBtn">Settings</button>
            <div id="chatConfigPane" style="display: none;"></div>

            <textarea id="chatInput"></textarea>
            <button id="microphoneBtn" class="attach-btn mic-btn">
                <i class="fa-solid fa-microphone"></i>
            </button>
            <input type="checkbox" id="chatThinkingToggle" checked />
            <button id="sendMsgBtn">Send</button>
            <button id="stopMsgBtn" style="display: none;">Stop</button>
            <div id="chatPane">
                <div id="messagesContainer">
                    <div id="messagesList"></div>
                </div>
                <div id="pendingImageTray"></div>
            </div>

            <!-- Parameters inside the settings pane -->
            <input type="text" id="globalSystemPrompt" value="You are a helpful assistant." />
            <input type="range" id="globalTemp" value="0.7" />
            <span id="valText-temp">0.7</span>
            <input type="range" id="globalTopP" value="0.9" />
            <span id="valText-topP">0.9</span>
            <input type="range" id="globalTopK" value="40" />
            <span id="valText-topK">40</span>
            <input type="range" id="globalMinP" value="0.05" />
            <span id="valText-minP">0.05</span>
            <input type="range" id="globalRepeatPenalty" value="1.1" />
            <span id="valText-repeatPenalty">1.1</span>
            <input type="number" id="globalMaxTokens" value="2048" />
            <input type="text" id="globalStop" value="" />
        `, {
            url: "http://localhost/",
            runScripts: "dangerously"
        });

        window = dom.window;

        // Mock window.document.addEventListener to prevent automatic initialization and timers
        window.document.addEventListener = jest.fn();

        // Mock window.fetch
        window.fetch = jest.fn();

        // Evaluate index.html's javascript inside the JSDOM window context
        window.eval(scriptCode);
    });

    afterEach(() => {
        jest.restoreAllMocks();
    });

    test("toggleChatConfigPane toggles display and active class", () => {
        const pane = window.document.getElementById('chatConfigPane');
        const btn = window.document.getElementById('toggleChatParamsBtn');

        // Initial state
        expect(pane.style.display).toBe('none');
        expect(btn.classList.contains('active')).toBe(false);

        // First toggle (should open)
        window.toggleChatConfigPane();
        expect(pane.style.display).toBe('flex');
        expect(btn.classList.contains('active')).toBe(true);

        // Second toggle (should close)
        window.toggleChatConfigPane();
        expect(pane.style.display).toBe('none');
        expect(btn.classList.contains('active')).toBe(false);
    });

    test("stopChatMessage aborts the active stream controller when running", () => {
        // Setup a mock abort controller
        const mockAbort = jest.fn();
        window.activeAbortController = {
            abort: mockAbort
        };

        // Call stopChatMessage
        window.stopChatMessage();

        // Verify abort was triggered
        expect(mockAbort).toHaveBeenCalled();
    });

    test("stopChatMessage does not crash if no active controller exists", () => {
        window.activeAbortController = null;

        expect(() => {
            window.stopChatMessage();
        }).not.toThrow();
    });

    test("buildChatRequestPayload sends the thinking toggle state", () => {
        const toggle = window.document.getElementById('chatThinkingToggle');

        expect(window.buildChatRequestPayload("hello", []).enable_thinking).toBe(true);

        toggle.checked = false;
        expect(window.buildChatRequestPayload("hello", []).enable_thinking).toBe(false);
    });

    test("microphone toggles recording and inserts the Whisper transcript without sending", async () => {
        const track = { stop: jest.fn() };
        Object.defineProperty(window.navigator, 'mediaDevices', {
            configurable: true,
            value: {
                getUserMedia: jest.fn().mockResolvedValue({
                    getTracks: () => [track]
                })
            }
        });

        class MockMediaRecorder {
            static latest = null;
            static isTypeSupported = jest.fn().mockReturnValue(true);

            constructor(stream, options) {
                this.stream = stream;
                this.mimeType = options.mimeType;
                this.state = 'inactive';
                this.listeners = {};
                MockMediaRecorder.latest = this;
            }

            addEventListener(name, callback) {
                this.listeners[name] = callback;
            }

            start() {
                this.state = 'recording';
            }

            stop() {
                this.state = 'inactive';
                const blob = new window.Blob(['voice'], { type: this.mimeType });
                this.listeners.dataavailable({ data: blob });
                this.finishPromise = this.listeners.stop();
            }
        }
        window.MediaRecorder = MockMediaRecorder;
        window.fetch
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    installed: true,
                    model_installed: true,
                    running: true,
                    model: 'small.en'
                })
            })
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({ text: 'Tell me a robot story.' })
            });

        await window.toggleMicrophoneRecording();

        const button = window.document.getElementById('microphoneBtn');
        expect(MockMediaRecorder.latest.state).toBe('recording');
        expect(button.classList.contains('recording')).toBe(true);
        expect(button.getAttribute('aria-pressed')).toBe('true');

        await window.toggleMicrophoneRecording();
        await MockMediaRecorder.latest.finishPromise;

        expect(track.stop).toHaveBeenCalled();
        expect(window.document.getElementById('chatInput').value).toBe(
            'Tell me a robot story.'
        );
        expect(button.classList.contains('recording')).toBe(false);
        expect(window.fetch).toHaveBeenCalledTimes(2);
        expect(window.fetch.mock.calls[1][0]).toBe('/api/speech/transcribe');
    });

    test("microphone falls back from a stale default to a concrete USB input", async () => {
        const stream = { getTracks: () => [] };
        const getUserMedia = jest.fn()
            .mockRejectedValueOnce(Object.assign(new Error('Requested device not found'), {
                name: 'NotFoundError'
            }))
            .mockResolvedValueOnce(stream);
        Object.defineProperty(window.navigator, 'mediaDevices', {
            configurable: true,
            value: {
                getUserMedia,
                enumerateDevices: jest.fn().mockResolvedValue([
                    { kind: 'audioinput', deviceId: 'default', label: 'Default' },
                    { kind: 'audioinput', deviceId: 'usb-mic', label: 'USB Audio Device Mono' }
                ])
            }
        });
        window.MediaRecorder = class {
            static isTypeSupported() { return true; }
            constructor() {
                this.mimeType = 'audio/webm;codecs=opus';
                this.state = 'inactive';
            }
            addEventListener() {}
            start() { this.state = 'recording'; }
        };
        window.fetch.mockResolvedValue({
            ok: true,
            json: async () => ({
                installed: true,
                model_installed: true,
                running: true,
                model: 'small.en'
            })
        });

        await window.toggleMicrophoneRecording();

        expect(getUserMedia).toHaveBeenCalledTimes(2);
        expect(getUserMedia.mock.calls[1][0].audio.deviceId).toEqual({
            exact: 'usb-mic'
        });
        expect(window.document.getElementById('microphoneBtn').classList.contains(
            'recording'
        )).toBe(true);
    });

    test("microphone uses local system capture when Chrome sees no inputs", async () => {
        Object.defineProperty(window.navigator, 'mediaDevices', {
            configurable: true,
            value: {
                getUserMedia: jest.fn().mockRejectedValue(
                    Object.assign(new Error('Requested device not found'), {
                        name: 'NotFoundError'
                    })
                ),
                enumerateDevices: jest.fn().mockResolvedValue([])
            }
        });
        window.MediaRecorder = class {};
        window.fetch
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    installed: true,
                    model_installed: true,
                    running: true,
                    model: 'small.en'
                })
            })
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({ recording: true, capture: 'system' })
            })
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({ text: 'Backend microphone worked.' })
            });

        await window.toggleMicrophoneRecording();

        const button = window.document.getElementById('microphoneBtn');
        expect(button.classList.contains('recording')).toBe(true);
        expect(window.fetch.mock.calls[1][0]).toBe('/api/speech/recording/start');

        await window.toggleMicrophoneRecording();

        expect(window.fetch.mock.calls[2][0]).toBe('/api/speech/recording/stop');
        expect(window.fetch.mock.calls[2][1].headers['X-LlamaStudio-Local']).toBe(
            'speech-capture'
        );
        expect(window.document.getElementById('chatInput').value).toBe(
            'Backend microphone worked.'
        );
    });

    test("handleChatImageFiles prepares a dropped raster image and renders a preview", async () => {
        const image = new window.File(
            [new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])],
            "small.png",
            { type: "image/png" }
        );

        await window.handleChatImageFiles([image]);

        const tray = window.document.getElementById('pendingImageTray');
        expect(tray.style.display).toBe('flex');
        expect(tray.querySelectorAll('.pending-image')).toHaveLength(1);
        expect(tray.querySelector('img').src).toMatch(/^data:image\/png;base64,/);
    });

    test("handleChatDrop accepts image files from the chat pane", async () => {
        const image = new window.File(
            [new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])],
            "dropped.png",
            { type: "image/png" }
        );
        const preventDefault = jest.fn();
        const pane = window.document.getElementById('chatPane');
        pane.classList.add('drag-active');

        await window.handleChatDrop({
            preventDefault,
            dataTransfer: { files: [image] }
        });

        expect(preventDefault).toHaveBeenCalled();
        expect(pane.classList.contains('drag-active')).toBe(false);
        expect(window.document.querySelectorAll('.pending-image')).toHaveLength(1);
    });

    test("handleChatMediaFiles prepares FLAC audio and renders an audio card", async () => {
        const audio = new window.File(
            [new Uint8Array([0x66, 0x4c, 0x61, 0x43, 0x00, 0x00, 0x00, 0x22])],
            "hello.flac",
            { type: "audio/flac" }
        );

        await window.handleChatMediaFiles([audio]);

        const tray = window.document.getElementById('pendingImageTray');
        expect(tray.style.display).toBe('flex');
        expect(tray.querySelectorAll('.pending-audio')).toHaveLength(1);
        expect(tray.textContent).toContain("hello.flac");
    });

    test("buildChatRequestPayload includes structured audio attachments", () => {
        const audio = {
            name: "hello.wav",
            mime_type: "audio/wav",
            data_url: "data:audio/wav;base64,UklGRg==",
            size: 4
        };

        const payload = window.buildChatRequestPayload("Transcribe this.", [], [audio]);

        expect(payload.audios).toEqual([audio]);
    });

    test("removePendingChatImage clears the prepared attachment", async () => {
        const image = new window.File(
            [new Uint8Array([0xff, 0xd8, 0xff, 0xd9])],
            "small.jpg",
            { type: "image/jpeg" }
        );
        await window.handleChatImageFiles([image]);

        window.removePendingChatImage(0);

        const tray = window.document.getElementById('pendingImageTray');
        expect(tray.style.display).toBe('none');
        expect(tray.querySelectorAll('.pending-image')).toHaveLength(0);
    });

    test("removePendingChatAudio clears the prepared attachment", async () => {
        const audio = new window.File(
            [new Uint8Array([0x66, 0x4c, 0x61, 0x43, 0x00, 0x00, 0x00, 0x22])],
            "hello.flac",
            { type: "audio/flac" }
        );
        await window.handleChatMediaFiles([audio]);

        window.removePendingChatAudio(0);

        const tray = window.document.getElementById('pendingImageTray');
        expect(tray.style.display).toBe('none');
        expect(tray.querySelectorAll('.pending-audio')).toHaveLength(0);
    });

    test("renderChatImages rejects non-raster data URLs", () => {
        const html = window.renderChatImages([{
            name: "unsafe.svg",
            data_url: "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4="
        }]);

        expect(html).toBe('');
    });

    test("renderChatAudios rejects non-audio data URLs", () => {
        const html = window.renderChatAudios([{
            name: "unsafe.html",
            data_url: "data:text/html;base64,PGgxPmJhZDwvaDE+"
        }]);

        expect(html).toBe('');
    });

    test("renderVisionRecovery offers the resolved projector download", () => {
        const html = window.renderVisionRecovery("Projector missing.", {
            status: "downloadable",
            repo_id: "author/model",
            filename: "mmproj-f16.gguf",
            model_path: "/models/model.gguf",
            size: 990000000
        });

        expect(html).toContain("Download projector");
        expect(html).toContain("mmproj-f16.gguf");
        expect(html).toContain("data-recovery=");
    });

    test("renderMessagesContent shows compact persisted response metrics", () => {
        window.renderMessagesContent([{
            role: "assistant",
            content: null,
            metrics: {
                prompt_tokens: 19,
                completion_tokens: 7,
                total_tokens: 26,
                cached_tokens: 5,
                elapsed_seconds: 0.21,
                prompt_seconds: 0.042,
                generation_seconds: 0.018,
                tokens_per_second: 33.3
            }
        }]);

        const metrics = window.document.querySelector('.response-metrics');
        expect(metrics).not.toBeNull();
        expect(metrics.textContent).toContain("26 tokens");
        expect(metrics.textContent).toContain("19 in / 7 out");
        expect(metrics.textContent).toContain("0.21s");
        expect(metrics.textContent).toContain("33.3 tok/s");
        expect(metrics.title).toContain("5 cached prompt tokens");
    });
});
