'use strict';

const vscode = require('vscode');
const { validate, DEFAULT_SETS } = require('./validate');

// Name of a SourceCluster : inline static const std::string Name = "SceneBuffers";
const NAME_RE = /static\s+(?:inline\s+)?(?:const|constexpr)\s+std::(?:string|string_view)\s+Name\s*=\s*"([A-Za-z_]\w*)"/g;

let scanned = new Set();
let scanToken = 0;
let diagnostics;
const timers = new Map();

function config() {
    return vscode.workspace.getConfiguration('mdl');
}

function knownSets() {
    const all = new Set(DEFAULT_SETS);
    for (const s of scanned) all.add(s);
    for (const s of config().get('extraSets', [])) all.add(s);
    return all;
}

// the scan is async and does not block the editor : it reads the C++ files of the workspace and takes the names of SourceClusters
// the globs are relative to the open folder , so the same extension works on any machine
async function scanSets() {
    const token = ++scanToken;
    const globs = config().get('sourceGlobs', ['**/include/**/*.hpp']);
    const found = new Set();

    try {
        for (const g of globs) {
            const files = await vscode.workspace.findFiles(g, '**/{node_modules,build,.git}/**', 3000);
            await Promise.all(files.map(async (uri) => {
                try {
                    const bytes = await vscode.workspace.fs.readFile(uri);
                    const text = Buffer.from(bytes).toString('utf8');
                    if (!text.includes('Name')) return;
                    for (const m of text.matchAll(NAME_RE)) found.add(m[1]);
                } catch { /* an unreadable file is not an error */ }
            }));
        }
    } catch { /* no workspace */ }

    if (token !== scanToken) return;   // a newer scan has started
    scanned = found;
    revalidateAll();
}

function toRange(doc, d) {
    const len = doc.lineAt(Math.min(d.line, doc.lineCount - 1)).text.length;
    return new vscode.Range(d.line, Math.min(d.start, len), d.line, Math.max(Math.min(d.end, len), Math.min(d.start, len) + 1));
}

function validateDocument(doc) {
    if (doc.languageId !== 'mdl') return;
    const diags = validate(doc.getText(), knownSets()).map((d) => {
        const diag = new vscode.Diagnostic(toRange(doc, d), d.message,
            d.severity === 'warning' ? vscode.DiagnosticSeverity.Warning : vscode.DiagnosticSeverity.Error);
        diag.source = 'MDL';
        return diag;
    });
    diagnostics.set(doc.uri, diags);
}

function scheduleValidate(doc) {
    if (doc.languageId !== 'mdl') return;
    const key = doc.uri.toString();
    clearTimeout(timers.get(key));
    timers.set(key, setTimeout(() => { timers.delete(key); validateDocument(doc); }, 250));
}

function revalidateAll() {
    for (const doc of vscode.workspace.textDocuments) validateDocument(doc);
}

function activate(context) {
    diagnostics = vscode.languages.createDiagnosticCollection('mdl');
    context.subscriptions.push(diagnostics);

    context.subscriptions.push(
        vscode.workspace.onDidOpenTextDocument(validateDocument),
        vscode.workspace.onDidChangeTextDocument((e) => scheduleValidate(e.document)),
        vscode.workspace.onDidCloseTextDocument((doc) => diagnostics.delete(doc.uri)),
        vscode.workspace.onDidChangeConfiguration((e) => {
            if (e.affectsConfiguration('mdl')) { scanSets(); }
        }),
        vscode.workspace.onDidSaveTextDocument((doc) => {
            // a new SourceCluster was written : the names have to be scanned again
            if (doc.fileName.endsWith('.hpp')) scanSets();
        }),
        vscode.commands.registerCommand('mdl.rescanSets', async () => {
            await scanSets();
            vscode.window.showInformationMessage(`MDL : ${knownSets().size} sets are known`);
        }),
    );

    revalidateAll();
    scanSets();   // async , the first validation uses the default list and is repeated when the scan ends
}

function deactivate() {
    for (const t of timers.values()) clearTimeout(t);
}

module.exports = { activate, deactivate };
