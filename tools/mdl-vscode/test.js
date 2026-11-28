'use strict';

// node test.js [folder with .mox files]
const fs = require('fs');
const path = require('path');
const { validate, DEFAULT_SETS } = require('./validate');

let failed = 0;
const check = (name, ok, extra = '') => { if (!ok) { failed++; console.log('FAIL', name, extra); } };

// 1. every real file has to be clean
const root = process.argv[2] || path.join(__dirname, '..', '..', 'Shaders');
let files = 0;
const walk = (dir) => {
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
        const p = path.join(dir, e.name);
        if (e.isDirectory()) walk(p);
        else if (e.name.endsWith('.mox')) {
            files++;
            const d = validate(fs.readFileSync(p, 'utf8'), DEFAULT_SETS);
            check(p, d.length === 0, JSON.stringify(d));
        }
    }
};
walk(root);
console.log(`real files : ${files}`);

// 2. bad inputs : every case has to give exactly the expected message part
const known = DEFAULT_SETS;
const bad = (name, text, part) => {
    const d = validate(text, known);
    check(name, d.some(x => x.message.includes(part)), JSON.stringify(d));
};
const ok = (name, text) => {
    const d = validate(text, known);
    check(name, d.length === 0, JSON.stringify(d));
};

bad('unknown set', '#include Nope<<0>>\n', 'does not exist');
bad('same index', '#include NKPD<<0>>\n#include NRC<<0>>\n', 'already used');
bad('included twice', '#include NKPD<<0>>\n#include NKPD<<1>>\n', 'already included');
bad('general not 0', '#include GENERAL<<1>>\n', 'index 0');
bad('wrong include', '#include NKPD\n', 'wrong #include');
bad('no set for {', '{\n}\n', 'has no set before it');
bad('} alone', '}\n', 'no opened set');
bad('set not included', '[NKPD]\n{\n}\n', 'no set "NKPD"');
bad('not closed', '#include NKPD<<0>>\n[NKPD]\n{\n', 'not closed');
bad('variable outside', '#include NKPD<<0>>\nStorageBuffer a<<ShaderRead>>\n', 'no set');
bad('unknown type', '#include NKPD<<0>>\n[NKPD]\n{\n Buffer a<<ShaderRead>>\n}\n', 'unknown type');
bad('unknown access', '#include NKPD<<0>>\n[NKPD]\n{\n StorageBuffer a<<ShaderReed>>\n}\n', 'unknown access');
bad('pp after access', '#include NKPD<<0>>\n[NKPD]\n{\n StorageBuffer a<<ShaderRead , CURRENT>>\n}\n', 'CURRENT or PREV');
bad('bad pp word', '#include NKPD<<0>>\n[NKPD]\n{\n StorageBuffer a<<CURENT , ShaderRead>>\n}\n', 'CURRENT or PREV');
bad('binding outside general', '#include NKPD<<0>>\n[NKPD]\n{\n StorageBuffer a<<3 , ShaderRead>>\n}\n', 'only in the GENERAL');
bad('general no binding', '#include GENERAL<<0>>\n[GENERAL]\n{\n StorageBuffer a<<ShaderRead>>\n}\n', 'binding first');
bad('duplicate binding', '#include GENERAL<<0>>\n[GENERAL]\n{\n StorageBuffer a<<0 , ShaderRead>>\n StorageBuffer b<<0 , ShaderRead>>\n}\n', 'binding 0 is already used');
bad('duplicate name', '#include NKPD<<0>>\n[NKPD]\n{\n StorageBuffer a<<ShaderRead>>\n StorageBuffer a<<ShaderRead>>\n}\n', 'already declared');
bad('name twice not pp', '#include NKPD<<0>>\n[NKPD]\n{\n StorageBuffer a<<ShaderRead>>\n StorageBuffer a<<CURRENT , ShaderRead>>\n}\n', 'only for a ping pong');
bad('empty declaration', '#include NKPD<<0>>\n[NKPD]\n{\n StorageBuffer a<<>>\n}\n', 'no declaration');
bad('too many tokens', '#include NKPD<<0>>\n[NKPD]\n{\n StorageBuffer a<<CURRENT , ShaderRead , x>>\n}\n', 'unsuitable format');
bad('garbage line', '#include NKPD<<0>>\n[NKPD]\n{\n hello\n}\n', 'undefined format');

ok('ping pong pair', '#include NKPD<<0>>\n[NKPD]\n{\n StorageImage a<<CURRENT , ShaderWrite>>\n StorageImage a<<PREV , ShaderRead>>\n}\n');
ok('general with binding', '#include GENERAL<<0>>\n[GENERAL]\n{\n StorageBuffer a<<0 , ShaderRead>>\n StorageImage b<<1 , CURRENT , ShaderBoth>>\n}\n');
ok('general without include', '[GENERAL]\n{\n Texture t<<0 , ShaderRead>>\n}\n');
ok('spaces and comments', '// hi\n#include   NKPD <<  0 >>\n\n[NKPD]\n{\n\tStorageBuffer   a  <<  ShaderRead  >>   \n}\r\n');
ok('no spaces around comma', '#include NKPD<<0>>\n[NKPD]\n{\n StorageImage a<<CURRENT,ShaderWrite>>\n}\n');

// the position of the error has to point into the line
const d = validate('#include NKPD<<0>>\n[NKPD]\n{\n StorageBuffer a<<ShaderReed>>\n}\n', known)[0];
check('position', d.line === 3 && d.start === 18 && d.end === 28, JSON.stringify(d));

console.log(failed ? `FAILED : ${failed}` : 'all tests passed');
process.exit(failed ? 1 : 0);
