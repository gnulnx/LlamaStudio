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

    test("renderChatImages rejects non-raster data URLs", () => {
        const html = window.renderChatImages([{
            name: "unsafe.svg",
            data_url: "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4="
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
