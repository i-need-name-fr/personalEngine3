# MDL - Mox Descriptor Language

Highlighting and validation of the `.mox` files ( MDL ) in VS Code .

## What is checked

- `#include Set<<index>>` : the set exists ( SourceCluster ) , the index is unique , GENERAL is the index 0 , no double include
- `[Set]` + `{ ... }` : the set is included above , the brackets are balanced , no set is left open
- `Type name<<...>>` : the type , the access , `CURRENT` / `PREV` before the access , no extra tokens
- the binding only in GENERAL and unique there , the same name twice only for a ping pong pair

The names of the sets are taken from the C++ sources of the open folder ( `static const std::string Name = "..."` in `**/include/**/*.hpp` ) , the scan is async .
The settings : `mdl.extraSets` , `mdl.sourceGlobs` . The command : `MDL : Rescan the sets of SourceClusters` .

## Install ( any machine )

```
code --install-extension mdl-validator-0.1.0.vsix
```

Rebuild after a change : `python build_vsix.py` . Test of the rules : `node test.js` .
The rules have to be the same as in `include/MDLCompiler/MDLCompiler.hpp` .
