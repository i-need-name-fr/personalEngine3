'use strict';

// the validator of MDL ( Mox Descriptor Language ) . it has NO dependency on vscode , so it can be tested alone ( node test.js ) .
// the rules have to be the same as in include/MDLCompiler/MDLCompiler.hpp ( _implCompileFile and _implProcessAttachmentDeclaration )

const TYPES = ['StorageBuffer', 'UniformBuffer', 'StorageImage', 'Texture', 'ConstantBuffer', 'SampledImage'];

const ACCESS = [
    'ShaderRead', 'ShaderWrite', 'ShaderBoth',
    'MemoryRead', 'MemoryWrite', 'MemoryBoth',
    'ColorRead', 'ColorWrite', 'ColorBoth',
    'DepthRead', 'DepthWrite', 'DepthBoth',
];

const PING_PONG = ['CURRENT', 'PREV'];
const GENERAL = 'GENERAL';
const GENERAL_INDEX = 0;

// the sets which exist in the engine even if the scan of the sources finds nothing
const DEFAULT_SETS = [
    GENERAL, 'AtmoScat', 'BloomSource', 'LightHub', 'SceneBuffers', 'SceneImages', 'SceneReSTIR',
    'SharcHeap', 'TextureHub', 'VolSource', 'VertexHeap', 'NKPD', 'NRC',
];

const IDENT = /^[A-Za-z_]\w*$/;

/**
 * @param {string} text the content of the file
 * @param {Iterable<string>} knownSets the names of the sets which exist ( SourceClusters )
 * @returns {{line:number,start:number,end:number,message:string,severity:'error'|'warning'}[]} line and columns are 0 based
 */
function validate(text, knownSets) {
    const diags = [];
    const known = new Set(knownSets);
    const add = (line, start, end, message, severity = 'error') => diags.push({ line, start, end, message, severity });

    const sets = new Map();           // name -> { index , line }
    const content = new Map();        // set name -> { keys : Set , names : Map<name , {pingPong}[]> , bindings : Map<binding , name> }
    let open = null;                  // { name , line }
    let braces = false;

    const lines = text.split(/\r?\n/);
    for (let i = 0; i < lines.length; i++) {
        const raw = lines[i].replace(/\s+$/, '');
        const lead = raw.length - raw.trimStart().length;
        const line = raw.trim();
        const whole = (message, severity) => add(i, lead, raw.length, message, severity);

        if (!line || line.startsWith('//')) continue;

        // ---- #include SetName<<index>>
        if (line.startsWith('#include')) {
            const m = /^#include\s+([^\s<]+)\s*<<\s*(\d+)\s*>>$/.exec(line);
            if (!m) {
                whole('wrong #include , expected : #include <SetName><<index>>');
                continue;
            }

            const name = m[1];
            const index = Number(m[2]);
            const nameStart = lead + line.indexOf(name, '#include'.length);
            const nameEnd = nameStart + name.length;

            if (!known.has(name)) {
                add(i, nameStart, nameEnd, `the set "${name}" does not exist , known sets : ${[...known].sort().join(' , ')}`);
            }

            if (name === GENERAL && index !== GENERAL_INDEX) {
                whole(`the general set has to have the index ${GENERAL_INDEX} , but it is ${index}`);
            }

            if (sets.has(name)) {
                add(i, nameStart, nameEnd, `the set "${name}" is already included ( line ${sets.get(name).line + 1} )`);
                continue;
            }

            let clash = null;
            for (const [other, data] of sets) {
                if (data.index === index) { clash = other; break; }
            }
            if (clash !== null) {
                whole(`the index ${index} of the set "${name}" is already used by the set "${clash}"`);
            }

            sets.set(name, { index, line: i });
            continue;
        }

        // ---- {
        if (line.startsWith('{')) {
            if (line !== '{') {
                whole('nothing is allowed after { on the same line');
            }
            if (!open) {
                whole('the bracket { has no set before it , expected : [SetName] and then {');
            } else if (braces) {
                whole(`the bracket { is opened twice , the set "${open.name}" is not closed`);
            }
            braces = true;
            continue;
        }

        // ---- }
        if (line.startsWith('}')) {
            if (line !== '}') {
                whole('nothing is allowed after } on the same line');
            }
            if (!open || !braces) {
                whole('the bracket } has no opened set , every } needs [SetName] and { before it');
            }
            braces = false;
            open = null;
            continue;
        }

        // ---- [SetName]
        if (line.startsWith('[')) {
            const m = /^\[([^\]]+)\]$/.exec(line);
            if (!m) {
                whole('wrong set line , expected : [SetName]');
                continue;
            }

            const name = m[1];
            if (open) {
                whole(`the set "${open.name}" is not closed with } , but the set [${name}] starts`);
            }

            if (!sets.has(name)) {
                if (name !== GENERAL) {
                    whole(`no set "${name}" was found , it has to be included above : #include ${name}<<index>>`);
                } else {
                    // GENERAL can be used without #include , it is always the index 0
                    let clash = null;
                    for (const [other, data] of sets) {
                        if (data.index === GENERAL_INDEX) { clash = other; break; }
                    }
                    if (clash !== null) {
                        whole(`the index ${GENERAL_INDEX} of the general set is already used by the set "${clash}"`);
                    }
                    sets.set(GENERAL, { index: GENERAL_INDEX, line: i });
                }
            }

            open = { name, line: i };
            braces = false;
            continue;
        }

        // ---- Type name<<declaration>>
        const m = /^(\S+)\s+([^\s<]+)\s*<<([^>]*)>>$/.exec(line);
        if (!m) {
            whole('undefined format , expected : Type name<<ACCESS>> , for example : StorageBuffer generalBuffer<<ShaderRead>>');
            continue;
        }

        const [, type, name, decl] = m;
        const typeStart = lead;
        const nameStart = lead + line.indexOf(name, type.length);
        const declStart = lead + line.indexOf('<<', type.length) + 2;

        if (!open || !braces) {
            whole('the variable has no set , it has to be inside [SetName] { ... }');
            continue;
        }

        if (!TYPES.includes(type)) {
            add(i, typeStart, typeStart + type.length, `unknown type "${type}" , expected : ${TYPES.join(' , ')}`);
        }

        if (!IDENT.test(name)) {
            add(i, nameStart, nameStart + name.length, `the name "${name}" is not an identifier`);
        }

        // the tokens of the declaration , with the columns
        const tokens = [];
        let offset = 0;
        for (const part of decl.split(',')) {
            const t = part.trim();
            const col = declStart + offset + (part.length - part.trimStart().length);
            tokens.push({ t, start: col, end: col + t.length });
            offset += part.length + 1;
        }

        if (decl.trim() === '') {
            add(i, declStart - 2, declStart + decl.length + 2, 'the variable has no declaration <<>> , expected : <<ACCESS>>');
            continue;
        }

        const isGeneral = open.name === GENERAL;
        let binding = null;
        let k = 0;

        if (isGeneral) {
            if (!/^\d+$/.test(tokens[0].t)) {
                add(i, tokens[0].start, tokens[0].end, 'the GENERAL set needs the binding first : <<binding , ACCESS>> or <<binding , CURRENT or PREV , ACCESS>>');
                continue;
            }
            binding = Number(tokens[0].t);
            k = 1;
        } else if (/^\d+$/.test(tokens[0].t)) {
            add(i, tokens[0].start, tokens[0].end, 'the binding is written only in the GENERAL set , the sets of SourceClusters have no binding');
            continue;
        }

        const rest = tokens.slice(k);
        let pingPong = null;
        let access = null;

        if (rest.length === 2) {
            if (!PING_PONG.includes(rest[0].t)) {
                add(i, rest[0].start, rest[0].end, `expected CURRENT or PREV , got "${rest[0].t}"`);
                continue;
            }
            pingPong = rest[0].t;
            access = rest[1];
        } else if (rest.length === 1) {
            access = rest[0];
        } else {
            const extra = rest[Math.min(rest.length - 1, 2)];
            add(i, extra.start, extra.end, 'unsuitable format of the declaration , expected : <<ACCESS>> or <<CURRENT or PREV , ACCESS>>' + (isGeneral ? ' with the binding first' : ''));
            continue;
        }

        if (!ACCESS.includes(access.t)) {
            const hint = PING_PONG.includes(access.t) ? ' ( CURRENT and PREV go BEFORE the access )' : '';
            add(i, access.start, access.end, `unknown access "${access.t}"${hint} , expected : ${ACCESS.join(' , ')}`);
            continue;
        }

        // ---- duplicates inside of the set
        if (!content.has(open.name)) content.set(open.name, { keys: new Set(), names: new Map(), bindings: new Map() });
        const c = content.get(open.name);

        const key = pingPong ? `${name}@${pingPong}` : name;
        if (c.keys.has(key)) {
            add(i, nameStart, nameStart + name.length, `the descriptor "${name}" is already declared in the set "${open.name}"`);
            continue;
        }

        const same = c.names.get(name) || [];
        if (same.length > 0 && !(pingPong && same.every(p => p))) {
            add(i, nameStart, nameStart + name.length, `the descriptor "${name}" is already declared in the set "${open.name}" , the same name twice is allowed only for a ping pong resource ( CURRENT and PREV )`);
            continue;
        }
        same.push(pingPong);
        c.names.set(name, same);
        c.keys.add(key);

        // the binding has to be unique only in GENERAL
        if (isGeneral) {
            if (c.bindings.has(binding)) {
                add(i, tokens[0].start, tokens[0].end, `the binding ${binding} is already used in the GENERAL set by "${c.bindings.get(binding)}"`);
            } else {
                c.bindings.set(binding, name);
            }
        }
    }

    if (open) {
        add(open.line, 0, lines[open.line].length, `the set "${open.name}" is not closed , the file ended without }`);
    }

    return diags;
}

module.exports = { validate, TYPES, ACCESS, PING_PONG, DEFAULT_SETS, GENERAL };
