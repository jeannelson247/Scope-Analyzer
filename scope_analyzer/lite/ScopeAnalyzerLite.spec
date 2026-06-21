# PyInstaller spec for ScopeAnalyzerLite. Build from repository root:
#   pyinstaller scope_analyzer/lite/ScopeAnalyzerLite.spec
import os

block_cipher = None
ROOT = os.path.abspath(os.path.join(SPECPATH, "..", ".."))

a = Analysis(
    [os.path.join(ROOT, 'scope_web/app_web.py')],
    pathex=[ROOT, os.path.join(ROOT, "scope_web")],
    datas=[

        (os.path.join(ROOT, "scope_web", "index.html"), "scope_web"),
        (os.path.join(ROOT, "presets.json"), "."),
        (os.path.join(ROOT, "docs"), "docs"),
        (os.path.join(ROOT, "examples"), "examples"),
    ],
    hiddenimports=['webview', 'webview.platforms.cocoa', 'scipy.signal', 'scipy.optimize', 'unittest', 'unittest.mock'],
    excludes=['tkinter', 'test', 'PySide6', 'mlx', 'mlx_lm', 'transformers'],
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ScopeAnalyzerLite',
    console=False,
    icon=None,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name='ScopeAnalyzerLite')
app = BUNDLE(
    coll,
    name='ScopeAnalyzerLite.app',
    bundle_identifier='org.scopeanalyzer.lite',
    info_plist={"NSHighResolutionCapable": True},
)
