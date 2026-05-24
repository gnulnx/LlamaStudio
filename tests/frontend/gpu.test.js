// Polyfill TextEncoder and TextDecoder for JSDOM in Jest environment
const { TextEncoder, TextDecoder } = require('util');
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

describe("Frontend GPU Detection and UI Estimator", () => {
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
        // Setup JSDOM
        dom = new JSDOM(`
            <div id="hfDetailsSidebar">
                <div id="hfSelectedModelRepo">google/gemma-2-9b-it-GGUF</div>
                <div id="hfSelectedModelName">gemma-2-9b-it-GGUF</div>
                
                <select id="hfQuantSelect">
                    <option value="gemma-2-9b-it.Q4_K_M.gguf">gemma-2-9b-it.Q4_K_M.gguf</option>
                    <option value="gemma-2-9b-it.Q8_0.gguf">gemma-2-9b-it.Q8_0.gguf</option>
                    <option value="gemma-2-9b-it.Huge.gguf">gemma-2-9b-it.Huge.gguf</option>
                </select>

                <!-- Detected GPU Info -->
                <div id="hfGpuInfo" style="display: none;">
                    <span id="hfGpuNameText"></span>
                </div>

                <!-- Smart VRAM / RTX 5090 Suitability Badges -->
                <div id="hfVramEstimator"></div>

                <button id="hfDownloadBtn">Download</button>
            </div>
        `, {
            url: "http://localhost/",
            runScripts: "dangerously"
        });

        window = dom.window;

        // Mock window.document.addEventListener to prevent automatic initialization and timers during evaluation
        window.document.addEventListener = jest.fn();

        // Mock window.fetch
        window.fetch = jest.fn();

        // Evaluate index.html's javascript inside the JSDOM window context naturally
        window.eval(scriptCode);

        // Define global variables expected by the template code AFTER evaluating to avoid overwrite
        window.selectedHfModelDetails = {
            id: "google/gemma-2-9b-it-GGUF",
            siblings: [
                { rfilename: "gemma-2-9b-it.Q4_K_M.gguf", size: 6000000000 }, // ~5.58 GB
                { rfilename: "gemma-2-9b-it.Q8_0.gguf", size: 10000000000 }, // ~9.31 GB
                { rfilename: "gemma-2-9b-it.Huge.gguf", size: 40000000000 }  // ~37.25 GB
            ]
        };
    });

    afterEach(() => {
        jest.restoreAllMocks();
    });

    test("fetchGpuInfo fetches from backend and updates the UI", async () => {
        // Mock successful GPU detection API call
        window.fetch.mockResolvedValueOnce({
            ok: true,
            json: async () => ({ name: "NVIDIA GeForce RTX 5090", vram: 31 })
        });

        // Trigger fetch
        await window.fetchGpuInfo();

        expect(window.fetch).toHaveBeenCalledWith('/api/gpu');
        expect(window.backendGpuInfo).toEqual({
            name: "NVIDIA GeForce RTX 5090",
            vram: 31
        });

        const gpuDiv = window.document.getElementById('hfGpuInfo');
        const gpuText = window.document.getElementById('hfGpuNameText');

        expect(gpuDiv.style.display).toBe('flex');
        expect(gpuText.textContent).toBe('NVIDIA GeForce RTX 5090 (31 GB VRAM)');
    });

    test("detectGPU falls back when backend info is missing", () => {
        window.backendGpuInfo = null;

        // Mock navigator.webgl or deviceMemory
        Object.defineProperty(window.navigator, 'deviceMemory', {
            value: 8,
            configurable: true
        });

        const gpu = window.detectGPU();
        expect(gpu.name).toBeDefined();
        expect(gpu.vram).toBeDefined();
    });

    test("detectGPU returns backend GPU details when present", () => {
        window.backendGpuInfo = {
            name: "AMD Radeon RX 7900 XTX",
            vram: 24
        };

        const gpu = window.detectGPU();
        expect(gpu).toEqual({
            name: "AMD Radeon RX 7900 XTX",
            vram: 24
        });
    });

    test("evaluateVramEstimate correctly calculates and displays Fits Fully badge for a 5.58 GB model on RTX 5090 (31 GB)", () => {
        window.backendGpuInfo = {
            name: "NVIDIA GeForce RTX 5090",
            vram: 31
        };

        // Select the Q4 model (~5.58 GB)
        window.document.getElementById('hfQuantSelect').value = "gemma-2-9b-it.Q4_K_M.gguf";

        window.evaluateVramEstimate();

        const badgeContainer = window.document.getElementById('hfVramEstimator');
        expect(badgeContainer.innerHTML).toContain("Fits Fully on NVIDIA GeForce RTX 5090");
        expect(badgeContainer.innerHTML).toContain("5.6 GB GGUF");
    });

    test("evaluateVramEstimate correctly displays Exceeds VRAM fallback badge when model exceeds VRAM limit", () => {
        window.backendGpuInfo = {
            name: "NVIDIA GeForce RTX 5090",
            vram: 31
        };

        // Select the Huge model (~37.25 GB, which exceeds 31 GB VRAM)
        window.document.getElementById('hfQuantSelect').value = "gemma-2-9b-it.Huge.gguf";

        window.evaluateVramEstimate();

        const badgeContainer = window.document.getElementById('hfVramEstimator');
        expect(badgeContainer.innerHTML).toContain("Exceeds 31GB VRAM - Heavy CPU Fallback");
        expect(badgeContainer.innerHTML).toContain("37.3 GB");
    });
});
