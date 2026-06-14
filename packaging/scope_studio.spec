# PyInstaller spec for Scope Studio - run FROM THE REPO ROOT on the
# TARGET OS (executables cannot be cross-compiled):
#   pip install pyinstaller
#   pyinstaller packaging/scope_studio.spec
# Output: dist/ScopeStudio (folder bundle; .app on macOS)

import os

block_cipher = None
ROOT = os.path.abspath(os.path.join(os.path.dirname(SPECPATH)))

a = Analysis(
    [os.path.join(ROOT, "app.py")],
    pathex=[ROOT],
    datas=[
        (os.path.join(ROOT, "presets.json"), "."),
        (os.path.join(ROOT, "style.qss"), "."),
        (os.path.join(ROOT, "docs"), "docs"),
        (os.path.join(ROOT, "examples"), "examples"),
    ],
    hiddenimports=[
        "scipy.signal", "scipy.optimize",
        "matplotlib.backends.backend_qtagg",
        "pyqtgraph.opengl",            # harmless if PyOpenGL absent
    ],
    excludes=["tkinter", "test", "unittest"],
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="ScopeStudio",
    console=False,
    icon=None,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name="ScopeStudio")

# macOS .app wrapper
app = BUNDLE(
    coll,
    name="ScopeStudio.app",
    bundle_identifier="org.scopestudio.app",
    info_plist={"NSHighResolutionCapable": True},
)
