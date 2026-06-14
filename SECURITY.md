# Security & Privacy

Scope Studio is built for laboratory data that is often unpublished and
sensitive. The design goal is simple to state: **your data never leaves
your machine.**

## Privacy model

- **No telemetry, no cloud calls.** The application makes no network
  requests except to a *local* Ollama server (`http://localhost:11434`)
  when — and only when — you use the AI side chat with the Ollama
  backend. The MLX direct and llama.cpp backends run fully in-process.
- **Original CSVs are read-only inputs.** Scope Studio records a file hash
  at load time; formulas, filters, reconstructions, extrapolations, and
  overlays are in-memory display/session state unless the user explicitly
  exports a new derived file.
- **Raw waveforms are never sent to any model.** The side chat receives
  only computed summary statistics, anomaly-scan text, and retrieved
  excerpts from paper folders you explicitly index. This is enforced in
  `ai_assistant.py` / `chat_actions.py` (grep for what is put into
  prompts — it is all derived text, never sample arrays).
- **Local models stay local.** Ollama and GGUF models execute on your
  hardware. Nothing in this codebase transmits prompts or data to
  external services.

## Code-execution surfaces (audited)

| Surface | Mitigation |
|---|---|
| Channel formula engine | AST-validated whitelist (`signal_tools._validate_formula`): no builtins, no attribute access, no imports, fixed function/constant table. The only `eval` in the codebase compiles this validated tree. |
| LLM output | Parsed JSON is dispatched against a **fixed tool table** in `chat_actions.run_tool` — model output is never executed as code, and unknown tool names are rejected. Ollama can additionally use structured outputs. |
| Runtime self-modification | Forbidden by project policy: AI may draft inactive tools in `tool_sandbox/drafts/`; humans review, test, promote, and log them. |
| Draft tool sandbox | `tool_sandbox.py` creates inactive templates with manifests and tests. Draft tools are not imported by the app and must run on in-memory arrays only. |
| Subprocess use | One call: `ollama list` with a fixed argument vector (no shell, no user input). |
| File parsing | CSV via pandas C parser (no pickle, no `eval`). PDFs (optional paper indexing) via pypdf — index only folders you trust. |

## Prompt-injection boundary

Treat CSV text, retrieved paper excerpts, and model replies as untrusted
instructions. They may influence what the model suggests, but they do not gain
authority over the numerical engine:

- The model can only request fixed actions or deterministic tools through the
  JSON action layer.
- Unknown tool names are rejected.
- Any formula suggested by the model is still evaluated by the AST sandbox:
  no imports, no attributes, no builtins, no scalar-only results.
- Raw waveform arrays are not placed in model prompts.

The test suite includes `tests/test_llm_action_safety.py` to guard this
boundary.

## For maintainers

- Pin dependency versions in `requirements.txt`; review upgrades.
- Never commit measurement data, model weights, or user profiles —
  `.gitignore` covers these; check before each push
  (`git status` should show code and docs only).
- New tools must go through the registry pattern: deterministic code,
  a self-test with ground truth, a development-log entry.

## Reporting a vulnerability

Open a GitHub issue with the label `security` (or contact the
maintainer privately for sensitive disclosures). Please include steps
to reproduce. Until a fix is released, workarounds will be documented
in the issue.
