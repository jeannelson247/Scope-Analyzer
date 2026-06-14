# Contributing to Scope Studio

Scope Studio is designed for students, researchers, and young scientists who
want publication-quality plots without rebuilding plotting standards from
scratch.

## Good First Contributions

- Add a journal style preset.
- Add a current monitor conversion preset.
- Improve CSV compatibility for a specific oscilloscope vendor.
- Add a translated UI/help string.
- Add a benchmark result for a local model on your device.

## Design Rules

- Keep numerical calculations deterministic and testable.
- Let the AI request only whitelisted actions.
- Keep user data local by default.
- Prefer readable scientific defaults over flashy UI.
- Document why a change helps the experimental workflow.

## Development Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Before Opening a Pull Request

```bash
python -m py_compile app.py ai_assistant.py chat_actions.py csv_loader.py \
  detect_anomalies.py nature_export.py paper_index.py signal_tools.py
python scripts/backtest_ui_iterations.py
```

Attach the generated screenshots from `backtests/` if your change affects the
interface or exported figure style.

