const { TextEncoder, TextDecoder } = require('util');
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const html = fs.readFileSync(path.resolve(__dirname, '../../app/templates/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1]
    .replace(/let conversations = \{\{ conversations \| tojson \}\};/, 'let conversations = [];')
    .replace(/let isServerRunning = \{% if server_running %\}true\{% else %\}false\{% endif %\};/, 'let isServerRunning = false;');

describe('Web Hugging Face discovery', () => {
    let dom, window, document;
    const response = (repo = 'org/model') => ({
        details: {
            id: repo,
            siblings: [
                {rfilename: 'mmproj-BF16.gguf', size: 1024 ** 3},
                {rfilename: 'model-BF16.gguf', size: 18 * 1024 ** 3},
                {rfilename: 'model-Q4_K_M.gguf', size: 6 * 1024 ** 3}
            ],
            gguf_sizes: {'model-BF16.gguf': 18 * 1024 ** 3, 'model-Q4_K_M.gguf': 6 * 1024 ** 3}
        },
        readme: '---\nlicense: apache-2.0\n---\n# Model card\nDocumentation from Hugging Face.'
    });

    beforeEach(() => {
        dom = new JSDOM(html, {url: 'http://localhost/', runScripts: 'outside-only'});
        window = dom.window;
        document = window.document;
        document.addEventListener = jest.fn();
        window.fetch = jest.fn();
        window.marked = {parse: jest.fn(text => `<p>${text}</p>`)};
        window.eval(script);
        window.backendGpuInfo = {name: 'RTX 5090', vram: 32};
    });
    afterEach(() => dom.window.close());

    test('Models settings tabs do not hide the Discover README', async () => {
        window.switchSettingsTab('load');
        window.fetch.mockResolvedValue({ok: true, json: async () => response()});
        await window.selectHfModelRow('org/model');
        const pane = document.getElementById('settingsTab-hfReadme');
        expect(pane.classList.contains('active')).toBe(true);
        expect(window.getComputedStyle(pane).display).toBe('flex');
        expect(window.marked.parse).toHaveBeenCalledWith('# Model card\nDocumentation from Hugging Face.');
        expect(document.getElementById('hfReadmeContent').textContent).toContain('Documentation from Hugging Face.');
        window.switchSettingsTab('inference');
        expect(window.getComputedStyle(pane).display).toBe('flex');
        expect(document.getElementById('tabBtn-hfReadme').classList.contains('active')).toBe(true);
    });

    test('chooses a model quantization instead of the vision projector and labels its GB size', async () => {
        window.fetch.mockResolvedValue({ok: true, json: async () => response()});
        await window.selectHfModelRow('org/model');
        const quant = document.getElementById('hfQuantSelect');
        expect(quant.options.length).toBe(2);
        expect(quant.value).toBe('model-Q4_K_M.gguf');
        expect(quant.selectedOptions[0].textContent).toContain('6.0 GB');
        expect(document.getElementById('hfVramEstimator').textContent).toContain('6.0 GB GGUF');
        expect(document.getElementById('hfDownloadBtn').disabled).toBe(false);
    });

    test('renders GB ranges and unknown sizes in the results column', async () => {
        window.selectHfModelRow = jest.fn();
        window.fetch.mockResolvedValue({ok: true, json: async () => ({models: [
            {id: 'org/model', gguf_sizes: response().details.gguf_sizes},
            {id: 'org/unknown'}
        ]})});
        await window.searchHfModels();
        const rows = document.querySelectorAll('#hfResultsTableBody tr');
        expect(rows[0].cells.length).toBe(6);
        expect(rows[0].cells[1].textContent).toBe('6.0–18.0 GB');
        expect(rows[1].cells[1].textContent).toBe('Unknown');
        expect(window.selectHfModelRow).toHaveBeenCalledWith('org/model');
    });

    test('a single quantization has one size instead of a duplicated range', () => {
        expect(window.formatHfModelSize({gguf_sizes: {'model.gguf': 5 * 1024 ** 3}})).toBe('5.0 GB');
    });

    test('clears the previous download selection while fetching another repository', async () => {
        window.fetch.mockResolvedValueOnce({ok: true, json: async () => response()});
        await window.selectHfModelRow('org/model');
        let finish;
        window.fetch.mockReturnValueOnce(new Promise(resolve => {finish = resolve;}));
        const pending = window.selectHfModelRow('org/next');
        expect(document.getElementById('hfDownloadBtn').disabled).toBe(true);
        expect(document.getElementById('hfQuantSelect').options.length).toBe(0);
        expect(window.selectedHfModelDetails).toBeNull();
        finish({ok: false});
        await pending;
        expect(document.getElementById('hfReadmeContent').textContent).toContain('Failed to retrieve details for org/next');
        expect(document.getElementById('hfDownloadBtn').disabled).toBe(true);
    });

    test('an older response cannot overwrite the selected model or README', async () => {
        let finish;
        window.fetch.mockReturnValueOnce(new Promise(resolve => {finish = resolve;}));
        const old = window.selectHfModelRow('org/old');
        window.fetch.mockResolvedValueOnce({ok: true, json: async () => response('org/new')});
        await window.selectHfModelRow('org/new');
        finish({ok: true, json: async () => response('org/old')});
        await old;
        expect(window.selectedHfModelDetails.id).toBe('org/new');
        expect(document.getElementById('hfSelectedModelRepo').textContent).toBe('org/new');
    });

    test('a repository containing only projectors cannot be downloaded as a model', async () => {
        const data = response();
        data.details.siblings = data.details.siblings.slice(0, 1);
        window.fetch.mockResolvedValue({ok: true, json: async () => data});
        await window.selectHfModelRow('org/model');
        expect(document.getElementById('hfDownloadBtn').disabled).toBe(true);
        expect(document.getElementById('hfVramEstimator').textContent).toContain('Repository has no GGUF files');
        expect(document.getElementById('hfReadmeContent').textContent).toContain('Documentation from Hugging Face.');
    });

    test('uses the backend model list to exclude calibration and draft files from quantizations', async () => {
        const data = response();
        data.details.siblings.push({rfilename: 'imatrix.gguf', size: 1000});
        data.details.gguf_files = ['model-BF16.gguf', 'model-Q4_K_M.gguf'];
        window.fetch.mockResolvedValue({ok: true, json: async () => data});
        await window.selectHfModelRow('org/model');
        expect(Array.from(document.getElementById('hfQuantSelect').options, o => o.value)).toEqual(data.details.gguf_files);
    });
});
