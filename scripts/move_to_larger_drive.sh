#!/usr/bin/env bash
# move_to_larger_drive.sh — safely move a folder from one external drive to
# another (copy -> verify -> confirm -> delete source). Never deletes
# anything until the copy is byte-verified AND you type YES.
#
# Step 1 — see your drives and exact names:
#     bash scripts/move_to_larger_drive.sh --list
#
# Step 2 — dry run (shows what WOULD copy, changes nothing):
#     bash scripts/move_to_larger_drive.sh --from "JEAN D2" --to "BIG DRIVE" --folder "Scope Studio Models"
#
# Step 3 — do it for real (add --go; you'll still confirm before any delete):
#     bash scripts/move_to_larger_drive.sh --from "JEAN D2" --to "BIG DRIVE" --folder "Scope Studio Models" --go
#
# Use --all instead of --folder to move every top-level item on the drive
# (system files like .Trashes/.Spotlight-V100/.fseventsd are skipped).
set -euo pipefail

FROM=""; TO=""; FOLDER="Scope Studio Models"; MOVE_ALL=0; GO=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) LIST=1; shift;;
    --from) FROM="${2:-}"; shift 2;;
    --to) TO="${2:-}"; shift 2;;
    --folder) FOLDER="${2:-}"; shift 2;;
    --all) MOVE_ALL=1; shift;;
    --go) GO=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

list_volumes() {
  echo "External volumes under /Volumes:"
  for p in /Volumes/*; do
    [[ -d "$p" ]] || continue
    df -k "$p" 2>/dev/null | awk -v name="$(basename "$p")" 'NR==2{
      printf "  %-30s total %7.1f GB   free %7.1f GB\n", name, $2/1048576, $4/1048576}'
  done
}

if [[ "${LIST:-0}" == 1 || ( -z "$FROM" && -z "$TO" ) ]]; then
  list_volumes
  echo
  echo "Re-run with: --from \"<smaller>\" --to \"<larger>\" [--folder \"<name>\" | --all]"
  exit 0
fi

SRC_ROOT="/Volumes/$FROM"; DST_ROOT="/Volumes/$TO"
[[ -d "$SRC_ROOT" ]] || { echo "source drive not found: $SRC_ROOT" >&2; exit 1; }
[[ -d "$DST_ROOT" ]] || { echo "dest drive not found:  $DST_ROOT" >&2; exit 1; }

EXCLUDES=(--exclude ".Spotlight-V100" --exclude ".Trashes" --exclude ".fseventsd" --exclude ".DocumentRevisions-V100")

if [[ "$MOVE_ALL" == 1 ]]; then
  SRC="$SRC_ROOT/"; DST="$DST_ROOT/"; LABEL="ALL of $FROM"
else
  SRC="$SRC_ROOT/$FOLDER"; DST="$DST_ROOT/"; LABEL="$FROM/$FOLDER"
  [[ -e "$SRC" ]] || { echo "folder not found: $SRC" >&2; exit 1; }
fi

echo "Plan: copy  [$LABEL]  ->  /Volumes/$TO/"
df -k "$SRC_ROOT" "$DST_ROOT" | awk 'NR==1||/Volumes/{print}'
echo

if [[ "$GO" != 1 ]]; then
  echo "DRY RUN (nothing is written). Items that would copy:"
  rsync -a -n --itemize-changes "${EXCLUDES[@]}" "$SRC" "$DST" | head -50
  echo "... add --go to copy for real."
  exit 0
fi

echo "Copying (rsync, resumable, progress)…"
rsync -a --info=progress2 "${EXCLUDES[@]}" "$SRC" "$DST"

echo "Verifying copy is complete (should list NOTHING below)…"
DIFF="$(rsync -a -n --itemize-changes "${EXCLUDES[@]}" "$SRC" "$DST")"
if [[ -n "$DIFF" ]]; then
  echo "VERIFY FAILED — these differ, NOT deleting source:" >&2
  echo "$DIFF" | head -50 >&2
  exit 1
fi
echo "Verified: destination matches source."

printf "Delete the source copy at [%s]? Type YES to confirm: " "$SRC"
read -r ans
if [[ "$ans" == "YES" ]]; then
  rm -rf "$SRC"
  echo "Done. Source removed; data now lives on $TO."
else
  echo "Kept the source (copy is on $TO too). Nothing deleted."
fi
