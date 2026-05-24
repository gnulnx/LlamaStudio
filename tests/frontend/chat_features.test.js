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
            <button id="sendMsgBtn">Send</button>
            <button id="stopMsgBtn" style="display: none;">Stop</button>
            <div id="messagesContainer">
                <div id="messagesList"></div>
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
});
