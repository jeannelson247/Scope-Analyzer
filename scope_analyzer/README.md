# Scope Analyzer builds

This folder is the release-control layer for the open-source app. The
scientific code stays in the repository root; each subfolder here is one
packaged product lane.

## Builds

- `lite/` - primary student/public app. Web UI, deterministic tools,
  no bundled model weights. Self-contained for users once built.
- `mac_mlx/` - Apple Silicon build with MLX runtime libraries bundled.
  Users still choose/download their own model folder.
- `classic/` - stable native PySide reference app (`app.py`) for the lab
  while the web UI matures.

## Data policy

All builds treat original CSV files as read-only measurements. Formulae,
filters, calibration, RLC reconstruction, and AI/tool outputs create
in-memory display traces or explicit exported copies only.

## Build expectation

Build on the target OS. End users should not need Python or pip; developers
need the build dependencies listed in each subfolder README.
