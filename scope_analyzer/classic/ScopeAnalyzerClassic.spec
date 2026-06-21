# PyInstaller spec for ScopeAnalyzerClassic. Build from repository root:
#   pyinstaller scope_analyzer/classic/ScopeAnalyzerClassic.spec
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
    [os.path.join(ROOT, 'app.py')],
    pathex=[ROOT, os.path.join(ROOT, "scope_web")],
    datas=[

        (os.path.join(ROOT, "presets.json"), "."),
        (os.path.join(ROOT, "style.qss"), "."),
        (os.path.join(ROOT, "docs"), "docs"),
        (os.path.join(ROOT, "examples"), "examples"),
    ],
    binaries=MLX_BINARIES,
    hiddenimports=['scipy.signal', 'scipy.optimize', 'matplotlib.backends.backend_qtagg', 'pyqtgraph.opengl'],
    excludes=['tkinter', 'test', 'unittest'],
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ScopeAnalyzerClassic',
    console=False,
    icon=None,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name='ScopeAnalyzerClassic')
app = BUNDLE(
    coll,
    name='ScopeAnalyzerClassic.app',
    bundle_identifier='org.scopeanalyzer.classic',
    info_plist={"NSHighResolutionCapable": True},
)
