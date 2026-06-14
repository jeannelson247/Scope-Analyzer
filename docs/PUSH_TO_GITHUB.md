# Putting Scope Studio on GitHub

A start-to-finish guide for a first-time publisher. Everything here runs in
the **macOS Terminal** (Spotlight → "Terminal"). No prior git experience
needed — copy/paste each block.

> Why a script and not "just run git"? An automated tool initialized a
> partial `.git` for you but could not finish it (the sandbox couldn't
> remove git's lock files). The script below wipes that partial repo and
> rebuilds it cleanly on your machine, where git works normally.

## 1. Make a free GitHub account (skip if you have one)

Go to <https://github.com> and sign up.

## 2. Initialize the repo locally

```bash
cd ~/Desktop/scope_studio03
bash scripts/setup_repo.sh
```

This removes any partial `.git`, re-initializes, and makes the first commit.
It prints a guard check — confirm `venv`, `raw shots`, `backups`, and
`__pycache__` all show **0** (those must never be published). It does **not**
upload anything yet.

To set your name/email on the commit, run it like:

```bash
GIT_AUTHOR_NAME="Your Name" GIT_AUTHOR_EMAIL="you@example.com" \
  bash scripts/setup_repo.sh
```

## 3. Create an empty repo on GitHub

On github.com click **New repository**. Name it e.g. `scope-studio`. **Do
not** add a README, license, or .gitignore — you already have them. Create
it, then copy the URL it shows, e.g.
`https://github.com/yourname/scope-studio.git`.

## 4. Connect and upload

```bash
git remote add origin https://github.com/yourname/scope-studio.git
git push -u origin main
```

If prompted to authenticate, GitHub will walk you through a browser login
(or use a personal access token). After this, your code is online.

## 5. Turn on the PulseLab web app (optional)

On your repo: **Settings ▸ Pages ▸ Source = "GitHub Actions"**. The included
`pulselab-pages` workflow then publishes the `pulselab/` web app at
`https://yourname.github.io/scope-studio/` on every push to `main`.

## 6. Build the desktop apps (optional)

Tag a version and push the tag to trigger the build workflow:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Under your repo's **Actions** tab you'll find downloadable builds: **Lite**
(Windows / macOS / Linux) and **Full** (macOS + MLX).

## Day-to-day after this

```bash
git add -A
git commit -m "describe what changed"
git push
```

## If git complains about a `.lock` file

```bash
rm -f .git/*.lock .git/objects/*.lock
```

Then retry your commit/push. (You can also re-run `setup_repo.sh` to rebuild
the repo from scratch — your files are never touched, only git's metadata.)
