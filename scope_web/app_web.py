"""app_web.py — desktop shell for the web-frontend Scope Studio.

Hosts index.html in a native window (pywebview) and exposes the real Python
backend to the page as `window.pywebview.api`. The browser is the view; Python
does the work (csv_loader, compute, MLX later).

Run:
    pip install pywebview        # one-time (uses the OS WebView; no Chromium)
    python scope_web/app_web.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend_api import Api  # noqa: E402


def main() -> int:
    try:
        import webview
    except ImportError:
        print("pywebview is not installed. Run:  pip install pywebview")
        return 1
    here = os.path.dirname(os.path.abspath(__file__))
    api = Api()
    window = webview.create_window(
        "Scope Studio — web", os.path.join(here, "index.html"),
        js_api=api, width=1400, height=900, min_size=(1000, 640),
        background_color="#1c1d21",
    )
    api.set_window(window)
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
