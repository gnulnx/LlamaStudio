// Application defaults and the immutable policy of the response in progress.
let sandboxDefaults = null;
let activeSandbox = null;
let sandboxGeneration = 0;
let sandboxSaving = false;

function renderSandboxSummary() {
    const element = document.getElementById('sandboxSummary');
    const settings = activeSandbox || sandboxDefaults;
    if (!element || !settings) return;
    const files = settings.disable_sandbox ? 'Files unrestricted' : 'Files restricted';
    const commands = settings.command_policy === 'allow' ? 'Commands allowed' : 'Commands blocked';
    const text = `Sandbox: ${settings.root} · ${files} · ${commands}`;
    element.textContent = text;
    element.title = `${text}${activeSandbox ? ' · Current response' : ' · Next request'}`;
}

function setActiveSandbox(settings) {
    activeSandbox = settings;
    renderSandboxSummary();
}

async function sandboxResponse(response) {
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Could not update settings.');
    return data;
}

async function loadSandboxSettings(populate = false) {
    if (sandboxSaving) return;
    const generation = ++sandboxGeneration;
    try {
        const settings = await sandboxResponse(await fetch('/api/settings/sandbox', {cache: 'no-store'}));
        if (generation !== sandboxGeneration) return;
        sandboxDefaults = settings;
        renderSandboxSummary();
        if (populate) {
            document.getElementById('sandboxRoot').value = sandboxDefaults.root;
            document.getElementById('sandboxRestricted').checked = !sandboxDefaults.disable_sandbox;
            document.getElementById('sandboxCommands').checked = sandboxDefaults.command_policy === 'allow';
        }
    } catch (error) {
        if (populate) showSandboxMessage(error.message, true);
        else if (!sandboxDefaults) document.getElementById('sandboxSummary').textContent = 'Sandbox: unavailable';
    }
}

function showSandboxMessage(message, error = false) {
    const element = document.getElementById('sandboxSettingsMessage');
    element.textContent = message;
    element.dataset.error = String(error);
}

async function saveSandboxSettings() {
    const button = document.getElementById('sandboxSave');
    button.disabled = true;
    sandboxSaving = true;
    ++sandboxGeneration;
    try {
        const payload = {
            root: document.getElementById('sandboxRoot').value.trim(),
            disable_sandbox: !document.getElementById('sandboxRestricted').checked,
            command_policy: document.getElementById('sandboxCommands').checked ? 'allow' : 'deny',
        };
        sandboxDefaults = await sandboxResponse(await fetch('/api/settings/sandbox', {
            method: 'PUT',
            headers: {'Content-Type': 'application/json', 'X-LlamaStudio-Settings': '1'},
            body: JSON.stringify(payload),
        }));
        document.getElementById('sandboxRoot').value = sandboxDefaults.root;
        renderSandboxSummary();
        showSandboxMessage('Saved. New requests use these settings; a response in progress keeps its original sandbox.');
    } catch (error) {
        showSandboxMessage(error.message, true);
    } finally {
        sandboxSaving = false;
        button.disabled = false;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    loadSandboxSettings();
    setInterval(() => loadSandboxSettings(), 3000);
});
