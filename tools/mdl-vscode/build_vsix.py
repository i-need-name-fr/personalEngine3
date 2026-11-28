"""Builds mdl-validator-<version>.vsix without node / vsce ( a .vsix is a zip with a manifest ) .
python build_vsix.py
install : code --install-extension mdl-validator-<version>.vsix
"""
import json
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["package.json", "extension.js", "validate.js", "language-configuration.json",
         os.path.join("syntaxes", "mdl.tmLanguage.json"), "README.md"]

pkg = json.load(open(os.path.join(HERE, "package.json"), encoding="utf-8"))
name, version, publisher = pkg["name"], pkg["version"], pkg["publisher"]
out = os.path.join(HERE, f"{name}-{version}.vsix")

content_types = """<?xml version="1.0" encoding="utf-8"?>
<Types>
  <Default Extension=".json" ContentType="application/json"/>
  <Default Extension=".js" ContentType="application/javascript"/>
  <Default Extension=".md" ContentType="text/markdown"/>
  <Default Extension=".vsixmanifest" ContentType="text/xml"/>
</Types>
"""

manifest = f"""<?xml version="1.0" encoding="utf-8"?>
<PackageManifest Version="2.0.0" xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011" xmlns:d="http://schemas.microsoft.com/developer/vsx-schema-design/2011">
  <Metadata>
    <Identity Language="en-US" Id="{name}" Version="{version}" Publisher="{publisher}"/>
    <DisplayName>{pkg["displayName"]}</DisplayName>
    <Description xml:space="preserve">{pkg["description"]}</Description>
    <Categories>{",".join(pkg.get("categories", []))}</Categories>
    <Properties>
      <Property Id="Microsoft.VisualStudio.Code.Engine" Value="{pkg["engines"]["vscode"]}" />
      <Property Id="Microsoft.VisualStudio.Code.ExtensionDependencies" Value="" />
      <Property Id="Microsoft.VisualStudio.Code.ExtensionPack" Value="" />
      <Property Id="Microsoft.VisualStudio.Code.ExtensionKind" Value="workspace" />
      <Property Id="Microsoft.VisualStudio.Code.LocalizedLanguages" Value="" />
    </Properties>
  </Metadata>
  <Installation>
    <InstallationTarget Id="Microsoft.VisualStudio.Code"/>
  </Installation>
  <Dependencies/>
  <Assets>
    <Asset Type="Microsoft.VisualStudio.Code.Manifest" Path="extension/package.json" Addressable="true" />
  </Assets>
</PackageManifest>
"""

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", content_types)
    z.writestr("extension.vsixmanifest", manifest)
    for f in FILES:
        z.write(os.path.join(HERE, f), "extension/" + f.replace(os.sep, "/"))

print("built", out)
