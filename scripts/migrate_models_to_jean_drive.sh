#!/usr/bin/env bash
# Migrate the local model vault from the small drive to the large drive.
#   SOURCE (contents): /Volumes/JEAN D2/Scope Studio Models
#   DEST:              /Volumes/JeanDrive1/Models   (created if missing)
#
# The CONTENTS of "Scope Studio Models" (the mlx/ and gguf/ folders) are
# copied into "Models", so you end up with /Volumes/JeanDrive1/Models/mlx/...
#
# Copy -> verify -> confirm -> delete source. Nothing is deleted until the
# copy is byte-verified AND you type YES.
#
#   Dry run (changes nothing):   bash scripts/migrate_models_to_jean_drive.sh
#   Do it for real:              bash scripts/migrate_models_to_jean_drive.sh --go
set -euo pipefail

SRC="/Volumes/JEAN D2/Scope Studio Models"
DST="/Volumes/JeanDrive1/Models"
SRC_DRIVE="/Volumes/JEAN D2"
DST_DRIVE="/Volumes/JeanDrive1"
GO=0; [[ "${1:-}" == "--go" ]] && GO=1

[[ -d "$SRC_DRIVE" ]] || { echo "Source drive not mounted: $SRC_DRIVE" >&2; exit 1; }
[[ -d "$DST_DRIVE" ]] || { echo "Large drive not mounted: $DST_DRIVE — plug it in and check the exact name in Finder." >&2; exit 1; }
[[ -d "$SRC" ]] || { echo "Source folder not found: $SRC" >&2; exit 1; }

mkdir -p "$DST"

need_k=$(du -sk "$SRC" | awk '{print $1}')
free_k=$(df -k "$DST_DRIVE" | awk 'NR==2{print $4}')
printf "Source size: %.1f GB   |   Free on %s: %.1f GB\n" \
  "$(awk "BEGIN{print $need_k/1048576}")" "$DST_DRIVE" \
  "$(awk "BEGIN{print $free_k/1048576}")"
[[ "$free_k" -gt "$need_k" ]] || { echo "Not enough free space on $DST_DRIVE." >&2; exit 1; }

EXC=(--exclude ".Spotlight-V100" --exclude ".Trashes" --exclude ".fseventsd" --exclude ".DocumentRevisions-V100")

if [[ "$GO" != 1 ]]; then
  echo
  echo "DRY RUN — would copy the CONTENTS of:"
  echo "    $SRC/"
  echo "into:"
  echo "    $DST/"
  echo "First items that would copy:"
  rsync -a -n --itemize-changes "${EXC[@]}" "$SRC/" "$DST/" | head -40
  echo "... re-run with --go to copy for real."
  exit 0
fi

echo "Copying (rsync, resumable, with progress)…"
# Use flags supported by the old rsync (2.6.9) that ships with macOS:
# --progress + --partial (NOT --info=progress2, which is rsync 3.1+).
rsync -a --partial --progress "${EXC[@]}" "$SRC/" "$DST/"

echo "Verifying the copy is complete (should print NOTHING below)…"
DIFF="$(rsync -a -n --itemize-changes "${EXC[@]}" "$SRC/" "$DST/")"
if [[ -n "$DIFF" ]]; then
  echo "VERIFY FAILED — source and dest differ, NOT deleting source:" >&2
  echo "$DIFF" | head -40 >&2
  exit 1
fi
echo "Verified: $DST matches $SRC exactly."

printf "Delete the source copy at [%s]? Type YES to confirm: " "$SRC"
read -r ans
if [[ "$ans" == "YES" ]]; then
  rm -rf "$SRC"
  echo "Done — models now live at $DST (source removed)."
  echo "Tip: point the app at the new vault if it doesn't auto-detect:"
  echo "    export SCOPE_STUDIO_EXTERNAL_MLX_MODELS=\"$DST/mlx\""
else
  echo "Kept the source. The verified copy is also at $DST."
fi
