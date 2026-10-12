const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');
const script = fs.readFileSync(path.resolve(__dirname, '../../app/static/sandbox.js'), 'utf8');

describe('Persistent sandbox settings', () => {
    let dom, window;
    const original = {root: '/original', disable_sandbox: false, command_policy: 'allow'};
    beforeEach(() => {
        dom = new JSDOM('<button id="sandboxSummary"></button><input id="sandboxRoot"><input type="checkbox" id="sandboxRestricted"><input type="checkbox" id="sandboxCommands"><button id="sandboxSave"></button><p id="sandboxSettingsMessage"></p>', {runScripts: 'dangerously'});
        window = dom.window;
        window.fetch = jest.fn().mockResolvedValue({ok: true, json: async () => original});
        window.setInterval = jest.fn();
        window.eval(script);
    });
    afterEach(() => dom.window.close());

    test('loads persisted settings into the form and header as plain text', async () => {
        const data = {...original, root: '/<img src=x onerror=alert(1)>'};
        window.fetch.mockResolvedValue({ok: true, json: async () => data});
        await window.loadSandboxSettings(true);
        expect(window.document.getElementById('sandboxRoot').value).toBe(data.root);
        expect(window.document.getElementById('sandboxSummary').textContent).toContain(data.root);
        expect(window.document.querySelector('img')).toBeNull();
        expect(window.document.getElementById('sandboxRestricted').checked).toBe(true);
    });

    test('saves explicit permissions while showing the active response snapshot', async () => {
        await window.loadSandboxSettings(true);
        window.setActiveSandbox(original);
        const next = {root: '/next', disable_sandbox: true, command_policy: 'deny'};
        window.document.getElementById('sandboxRoot').value = next.root;
        window.document.getElementById('sandboxRestricted').checked = false;
        window.document.getElementById('sandboxCommands').checked = false;
        window.fetch.mockImplementation(async (url, options) => ({ok: true, json: async () => options?.method === 'PUT' ? next : original}));
        await window.saveSandboxSettings();
        const save = window.fetch.mock.calls.find(([, options]) => options?.method === 'PUT');
        expect(JSON.parse(save[1].body)).toEqual(next);
        expect(save[1].headers['X-LlamaStudio-Settings']).toBe('1');
        expect(window.document.getElementById('sandboxSummary').textContent).toContain('/original');
        window.setActiveSandbox(null);
        expect(window.document.getElementById('sandboxSummary').textContent).toContain('/next');
        expect(window.document.getElementById('sandboxSummary').textContent).toContain('Commands blocked');
    });

    test('validation failures preserve the effective header and expose the actual error', async () => {
        await window.loadSandboxSettings(true);
        window.fetch.mockResolvedValue({ok: false, json: async () => ({detail: 'Directory does not exist'})});
        await window.saveSandboxSettings();
        expect(window.document.getElementById('sandboxSettingsMessage').textContent).toBe('Directory does not exist');
        expect(window.document.getElementById('sandboxSave').disabled).toBe(false);
        expect(window.document.getElementById('sandboxSummary').textContent).toContain('/original');
    });
});
