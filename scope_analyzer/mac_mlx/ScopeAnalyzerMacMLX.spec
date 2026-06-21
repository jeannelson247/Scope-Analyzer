# PyInstaller spec for ScopeAnalyzerMacMLX. Build from repository root:
#   pyinstaller scope_analyzer/mac_mlx/ScopeAnalyzerMacMLX.spec
import os

block_cipher = None
ROOT = os.path.abspath(os.path.join(SPECPATH, "..", ".."))
MLX_LIB = os.path.join(ROOT, "venv", "lib", "python3.14", "site-packages", "mlx", "lib")
MLX_BINARIES = [
    (os.path.join(MLX_LIB, name), "mlx/lib")
    for name in ("libjaccl.dylib",)
    if os.path.exists(os.path.join(MLX_LIB, name))
]

a = Analysis(
    [os.path.join(ROOT, 'scope_web/app_web.py')],
    pathex=[ROOT, os.path.join(ROOT, "scope_web")],
    datas=[

        (os.path.join(ROOT, "scope_web", "index.html"), "scope_web"),
        (os.path.join(ROOT, "presets.json"), "."),
        (os.path.join(ROOT, "docs"), "docs"),
        (os.path.join(ROOT, "examples"), "examples"),
    ],
    binaries=MLX_BINARIES,
    hiddenimports=['webview', 'webview.platforms.cocoa', 'scipy.signal', 'scipy.optimize', 'mlx', 'mlx_lm', 'huggingface_hub', 'hf_xet'],
    excludes=['tkinter', 'test', 'unittest', 'PySide6'],
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ScopeAnalyzerMacMLX',
    console=False,
    icon=None,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name='ScopeAnalyzerMacMLX')
app = BUNDLE(
    coll,
    name='ScopeAnalyzerMacMLX.app',
    bundle_identifier='org.scopeanalyzer.macmlx',
    info_plist={"NSHighResolutionCapable": True},
)
