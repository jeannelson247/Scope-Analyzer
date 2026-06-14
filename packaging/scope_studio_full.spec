# PyInstaller spec for the FULL (Mac-optimized) Scope Studio build.
# Build ONLY on macOS, in a venv that has BOTH requirements.txt and
# requirements-mlx-mac.txt installed:
#   pip install -r requirements.txt -r requirements-mlx-mac.txt pyinstaller
#   pyinstaller packaging/scope_studio_full.spec
# Output: dist/ScopeStudioFull.app
#
# Difference from scope_studio.spec: bundles the MLX direct backend
# (mlx, mlx_lm) so Apple Silicon users get the in-process MLX path
# without a separate pip install. Everything else (presets, docs,
# examples, deterministic tools) is identical to the Lite/base build.

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
        "pyqtgraph.opengl",
        "mlx", "mlx_lm",            # MLX direct backend (Apple Silicon only)
        "huggingface_hub", "hf_xet",
    ],
    excludes=["tkinter", "test", "unittest"],
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="ScopeStudioFull",
    console=False,
    icon=None,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name="ScopeStudioFull")

app = BUNDLE(
    coll,
    name="ScopeStudioFull.app",
    bundle_identifier="org.scopestudio.full",
    info_plist={"NSHighResolutionCapable": True},
)
