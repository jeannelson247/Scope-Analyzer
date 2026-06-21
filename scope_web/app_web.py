"""app_web.py - desktop shell for the web Scope Analyzer app.

Hosts scope_web/index.html in a native WebView and exposes the tested
Python backend as window.pywebview.api.
"""
from __future__ import annotations

import os
import sys

def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROOT = getattr(sys, "_MEIPASS", _repo_root())
sys.path.insert(0, os.path.join(ROOT, "scope_web"))
sys.path.insert(0, ROOT)

from backend_api import Api  # noqa: E402


def resource_path(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def main() -> int:
    try:
        import webview
    except ImportError:
        print("pywebview is not installed. For development run: pip install pywebview")
        return 1
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
