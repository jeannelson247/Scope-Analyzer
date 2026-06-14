"""
adaptive_profile.py - safe preference logging for Scope Studio.

The local AI is allowed to propose whitelisted UI actions, not rewrite source
code. This module records accepted actions and device/profile context so future
versions can suggest better defaults for a lab, user, instrument, or OS.
"""
from __future__ import annotations

import json
import os
import platform
import time
from typing import Any


PROFILE_FILENAME = "scope_studio_user_profile.json"


def profile_path(base_dir: str) -> str:
    return os.path.join(base_dir, PROFILE_FILENAME)


def device_summary() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def load_profile(base_dir: str) -> dict[str, Any]:
    try:
        with open(profile_path(base_dir), encoding="utf-8") as f:
            profile = json.load(f)
    except Exception:
        profile = {}
    profile.setdefault("device", device_summary())
    profile.setdefault("events", [])
    profile.setdefault("preferences", {})
    return profile


def save_profile(base_dir: str, profile: dict[str, Any]):
    with open(profile_path(base_dir), "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
        f.write("\n")


def record_event(base_dir: str, kind: str, payload: dict[str, Any]):
    profile = load_profile(base_dir)
    profile["events"].append({
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "kind": kind,
        "payload": payload,
    })
    profile["events"] = profile["events"][-200:]
    save_profile(base_dir, profile)

