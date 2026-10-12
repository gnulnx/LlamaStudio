const { TextEncoder, TextDecoder } = require('util');
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

// Exercise the functions served by the real template, including text-safe errors.
const html = fs.readFileSync(path.resolve(__dirname, '../../app/templates/index.html'), 'utf8');
const start = html.indexOf('    async function refreshInferenceSetup()');
const end = html.indexOf('    // ─── INITIALIZATION', start);
const script = html.slice(start, end);

describe('Local inference setup', () => {
    let dom, window;
    beforeEach(() => {
        dom = new JSDOM('<aside id="inferenceSetupCard" hidden><p id="inferenceSetupMessage"></p><button id="inferenceSetupButton"></button></aside>', {runScripts: 'dangerously'});
        window = dom.window;
        window.fetch = jest.fn();
        window.setTimeout = jest.fn();
        window.eval(script);
    });
    afterEach(() => dom.window.close());

    test('shows a setup action for an unusable engine without starting downloads', async () => {
        window.fetch.mockResolvedValue({ok: true, json: async () => ({usable: false, problem: 'Run lls setup'})});
        await window.refreshInferenceSetup();
        expect(window.document.getElementById('inferenceSetupCard').hidden).toBe(false);
        expect(window.document.getElementById('inferenceSetupMessage').textContent).toBe('Run lls setup');
        expect(window.fetch).toHaveBeenCalledTimes(1);
        expect(window.fetch).toHaveBeenCalledWith('/api/inference/status');
    });
    test('hides the card when a working engine is available', async () => {
        window.fetch.mockResolvedValue({ok: true, json: async () => ({usable: true})});
        await window.refreshInferenceSetup();
        expect(window.document.getElementById('inferenceSetupCard').hidden).toBe(true);
    });
    test('explicit setup action posts once, disables the button and polls progress', async () => {
        window.fetch.mockResolvedValueOnce({ok: true, json: async () => ({running: true})});
        window.fetch.mockResolvedValueOnce({ok: true, json: async () => ({running: true, message: 'Downloading'})});
        await window.setupLocalInference();
        expect(window.fetch).toHaveBeenCalledWith('/api/inference/setup', {method: 'POST', headers: {'X-LlamaStudio-Setup': '1'}});
        expect(window.document.getElementById('inferenceSetupButton').disabled).toBe(true);
        expect(window.document.getElementById('inferenceSetupMessage').textContent).toBe('Downloading');
        expect(window.setTimeout).toHaveBeenCalledTimes(1);
    });
    test('failed setup shows actionable text and enables retry', async () => {
        const message = '<script>bad</script> lls doctor';
        window.fetch.mockResolvedValueOnce({ok: false, json: async () => ({detail: message})});
        await window.setupLocalInference();
        expect(window.document.getElementById('inferenceSetupMessage').textContent).toBe(message);
        expect(window.document.querySelector('script')).toBeNull();
        expect(window.document.getElementById('inferenceSetupButton').disabled).toBe(false);
    });
});
