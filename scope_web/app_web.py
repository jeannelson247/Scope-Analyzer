"""app_web.py - desktop shell for the web Scope Analyzer app.

Hosts scope_web/index.html in a native WebView and exposes the tested
Python backend as window.pywebview.api.
"""
from __future__ import annotations

import filecmp
import os
import shutil
import sys

def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _resource_root() -> str:
    base = os.path.abspath(getattr(sys, "_MEIPASS", _repo_root()))
    candidates = [base]
    if getattr(sys, "frozen", False):
        candidates.append(os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "Resources"))
    candidates.append(os.path.join(os.path.dirname(base), "Resources"))
    for candidate in dict.fromkeys(os.path.abspath(c) for c in candidates):
        if os.path.exists(os.path.join(candidate, "scope_web", "index.html")):
            return candidate
    return base


ROOT = _resource_root()
sys.path.insert(0, os.path.join(ROOT, "scope_web"))
sys.path.insert(0, ROOT)

from backend_api import Api  # noqa: E402


def resource_path(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def ensure_user_examples() -> str | None:
    """Sync bundled examples to a user-writable folder on launch.

    Packaged resources live inside the .app bundle, which is awkward for the
    normal Open CSV workflow. This mirrors the tutorial/benchmark examples to
    ~/Documents/Scope Analyzer/examples and refreshes changed bundled files on
    later launches. Non-fatal; the in-app Examples menu can still load the
    bundle copy if the sync fails.
    """
    src = resource_path("examples")
    dst = os.path.join(os.path.expanduser("~"), "Documents", "Scope Analyzer", "examples")
    try:
        if not os.path.isdir(src):
            return dst if os.path.isdir(dst) else None
        os.makedirs(dst, exist_ok=True)
        for base, _dirs, files in os.walk(src):
            rel = os.path.relpath(base, src)
            out_dir = dst if rel == "." else os.path.join(dst, rel)
            os.makedirs(out_dir, exist_ok=True)
            for name in files:
                src_file = os.path.join(base, name)
                dst_file = os.path.join(out_dir, name)
                if (not os.path.exists(dst_file) or
                        not filecmp.cmp(src_file, dst_file, shallow=False)):
                    shutil.copy2(src_file, dst_file)
        return dst
    except Exception:
        return None  # bundled examples still load via the in-app Examples menu


def main() -> int:
    try:
        import webview
    except ImportError:
        print("pywebview is not installed. For development run: pip install pywebview")
        return 1
    ensure_user_examples()
    api = Api()
    window = webview.create_window(
        "Scope Analyzer - Lite", resource_path("scope_web", "index.html"),
        js_api=api, width=1400, height=900, min_size=(1000, 640),
        background_color="#1c1d21",
    )
    api.set_window(window)
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
