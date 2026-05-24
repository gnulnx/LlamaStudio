// Polyfill TextEncoder and TextDecoder for JSDOM in Jest environment
const { TextEncoder, TextDecoder } = require('util');
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

describe("Frontend Model Deletion Feature", () => {
    let scriptCode;
    let dom;
    let window;

    beforeAll(() => {
        const htmlPath = path.resolve(__dirname, '../../app/templates/index.html');
        const htmlContent = fs.readFileSync(htmlPath, 'utf8');
        const scriptMatch = htmlContent.match(/<script>([\s\S]*?)<\/script>/);
        if (!scriptMatch) {
            throw new Error("Could not find <script> block in index.html");
        }
        let rawScript = scriptMatch[1];
        rawScript = rawScript
            .replace(/let conversations = \{\{ conversations \| tojson \}\};/, 'let conversations = [];')
            .replace(/let loadedModelPath = '\{\{ current_model or "" \}\}';/, 'let loadedModelPath = "";')
            .replace(/let loadedModelName = '\{\{ current_model_name or "" \}\}';/, 'let loadedModelName = "";')
            .replace(/let isServerRunning = \{% if server_running %\}true\{% else %\}false\{% endif %\};/, 'let isServerRunning = false;');
        scriptCode = rawScript;
    });

    beforeEach(() => {
        dom = new JSDOM(`
            <div id="configSidebar" style="display: block;">
                <button id="sidebarDeleteBtn">Delete</button>
            </div>
        `, {
            url: "http://localhost/",
            runScripts: "dangerously"
        });

        window = dom.window;
        window.document.addEventListener = jest.fn();
        window.fetch = jest.fn();
        
        // Define confirm and alert as configurable, writable mocks to override JSDOM defaults
        Object.defineProperty(window, 'confirm', {
            value: jest.fn(),
            configurable: true,
            writable: true
        });
        Object.defineProperty(window, 'alert', {
            value: jest.fn(),
            configurable: true,
            writable: true
        });
        window.eval(scriptCode);

        window.rescanModels = jest.fn();
    });

    afterEach(() => {
        jest.restoreAllMocks();
    });

    test("triggerSelectedModelDelete deletes model successfully", async () => {
        window.selectedModelPath = "/home/user/.lmstudio/models/test_model.gguf";
        window.confirm.mockReturnValue(true);
        window.fetch.mockResolvedValueOnce({
            ok: true,
            json: async () => ({ status: "ok" })
        });

        await window.triggerSelectedModelDelete();

        expect(window.confirm).toHaveBeenCalledWith(
            expect.stringContaining("Are you sure you want to delete \"test_model.gguf\"?")
        );
        expect(window.fetch).toHaveBeenCalledWith('/api/models/delete', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: "/home/user/.lmstudio/models/test_model.gguf" })
        });

        const sidebar = window.document.getElementById('configSidebar');
        expect(sidebar.style.display).toBe('none');
        expect(window.selectedModelPath).toBe('');
        expect(window.rescanModels).toHaveBeenCalled();
        expect(window.alert).toHaveBeenCalledWith("Model successfully deleted from disk.");
    });

    test("triggerSelectedModelDelete does nothing if user cancels confirmation", async () => {
        window.selectedModelPath = "/home/user/.lmstudio/models/test_model.gguf";
        window.confirm.mockReturnValue(false);

        await window.triggerSelectedModelDelete();

        expect(window.confirm).toHaveBeenCalled();
        expect(window.fetch).not.toHaveBeenCalled();
        expect(window.selectedModelPath).toBe("/home/user/.lmstudio/models/test_model.gguf");
    });

    test("triggerSelectedModelDelete alerts user on backend failure", async () => {
        window.selectedModelPath = "/home/user/.lmstudio/models/test_model.gguf";
        window.confirm.mockReturnValue(true);
        window.fetch.mockResolvedValueOnce({
            ok: false,
            json: async () => ({ detail: "Cannot delete a model that is currently loaded." })
        });

        await window.triggerSelectedModelDelete();

        expect(window.fetch).toHaveBeenCalled();
        expect(window.alert).toHaveBeenCalledWith(
            "Deletion failed: Cannot delete a model that is currently loaded."
        );
        expect(window.selectedModelPath).toBe("/home/user/.lmstudio/models/test_model.gguf");
    });
});
