#!/usr/bin/env bash
# setup_repo.sh — initialize the Scope Studio git repository cleanly and make
# the first commit. Run this in your Mac Terminal from the project folder:
#
#     cd ~/Desktop/scope_studio03
#     bash scripts/setup_repo.sh
#
# It does NOT push (that needs your GitHub account). It prints the exact
# push commands at the end. Safe to re-run.
set -euo pipefail

# 0. sanity: are we in the project root?
if [[ ! -f app.py || ! -f requirements.txt ]]; then
  echo "Run this from the scope_studio03 project root (app.py not found)." >&2
  exit 1
fi

# 1. remove any partial/locked .git left by an automated tool, start clean
if [[ -d .git ]]; then
  echo "Removing existing .git (starting fresh)…"
  rm -rf .git
fi

# 2. init + identity (edit these to taste; they only affect commit authorship)
git init -q
git branch -M main
git config user.name  "${GIT_AUTHOR_NAME:-Jean Nelson}"
git config user.email "${GIT_AUTHOR_EMAIL:-thesquad693693@gmail.com}"

# 3. stage everything the .gitignore allows, then show what will be committed
git add -A
echo
echo "About to commit $(git diff --cached --name-only | wc -l | tr -d ' ') files."
echo "Guard check (all should be 0):"
echo "  venv:        $(git ls-files | grep -c '^venv/' || true)"
echo "  raw shots:   $(git ls-files | grep -c 'T0.*\.CSV' || true)"
echo "  backups:     $(git ls-files | grep -c '^backups/' || true)"
echo "  __pycache__: $(git ls-files | grep -c '__pycache__' || true)"
echo

# 4. first commit
git commit -q -m "Initial commit: Scope Studio (PulseLab web + Lite + Full desktop)"
echo "Committed. History:"
git log --oneline -1
echo
cat <<'NEXT'
──────────────────────────────────────────────────────────────────────
Next steps (need your GitHub account):

1. Create an EMPTY repo on github.com (no README/license/.gitignore —
   you already have them). Copy its URL, e.g.
   https://github.com/<you>/scope-studio.git

2. Connect and push:
     git remote add origin https://github.com/<you>/scope-studio.git
     git push -u origin main

3. (PulseLab web app) On GitHub: Settings ▸ Pages ▸ Source = "GitHub
   Actions". The pulselab-pages workflow then deploys pulselab/ on push.

4. (Desktop builds) Tag a release to trigger the build matrix:
     git tag v0.1.0
     git push origin v0.1.0
   Lite (all OSes) and Full (Mac/MLX) artifacts appear under Actions.
──────────────────────────────────────────────────────────────────────
NEXT
