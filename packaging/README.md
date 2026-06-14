# Packaging Scope Studio

Executables must be built **on the OS they target** — there is no
cross-compiling a Windows .exe from a Mac. The spec file makes each
build a two-liner.

## macOS (your machine)

```bash
cd ~/Desktop/scope_studio03
source venv/bin/activate
pip install pyinstaller
pyinstaller packaging/scope_studio.spec
open dist/ScopeStudio.app        # test it
```

Notes:
- First build takes a few minutes; output is `dist/ScopeStudio.app`
  (a few hundred MB - PySide6 and scipy dominate).
- Distribution outside your own machine needs codesigning +
  notarization (Apple Developer ID) or users must right-click > Open.
- Models are NOT bundled: the app talks to whatever Ollama has
  installed. This keeps the bundle small and the licenses clean.

## Windows (for the PXI machine)

Same two commands in a Windows venv (`pip install -r requirements.txt
pyinstaller`, then `pyinstaller packaging\scope_studio.spec`).
Output: `dist\ScopeStudio\ScopeStudio.exe`. Build it on the PXI
controller or any Windows box with the same Python minor version.

## Linux

Identical procedure; output runs on distros with comparable glibc.
For broad distribution prefer an AppImage (pyinstaller output wrapped
with appimagetool) - revisit when there is demand.

## CI later (when the repo is public)

A GitHub Actions matrix (macos-latest / windows-latest /
ubuntu-latest) can run this spec on each OS per release tag and attach
the three bundles to the release - that is the long-term answer to
"self-contained on every device" without owning every machine.

## Checklist before each release

1. `python3 -m py_compile *.py scripts/*.py`
2. `python3 scripts/backtest_real_data.py <known shot> --expect-amps ...`
3. `python3 scripts/benchmark_data_analysis.py --mock`
4. Update DEVELOPMENT_LOG + version note in README.
