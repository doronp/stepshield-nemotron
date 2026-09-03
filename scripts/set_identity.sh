#!/usr/bin/env bash
# Set the release identity (GitHub owner + HuggingFace namespace) in one step.
#
# The committed artifacts deliberately contain no personal or organizational
# handle; every repo/model URL uses the placeholder tokens __GH_OWNER__ and
# __HF_OWNER__. A maintainer runs this ONCE at publish time:
#
#   bash scripts/set_identity.sh <github-owner> <hf-namespace> [extra paths...]
#
# Example:
#   bash scripts/set_identity.sh myorg myorg ../stepshield-nemotron-hf
#
# It rewrites every tracked text file in this repo (and any extra paths given,
# e.g. the HuggingFace upload directory or article drafts), then prints a grep
# so you can verify no token remains.
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <github-owner> <hf-namespace> [extra files-or-dirs...]" >&2
  exit 1
fi
GH_OWNER="$1"; HF_OWNER="$2"; shift 2
case "$GH_OWNER$HF_OWNER" in
  *[!A-Za-z0-9._-]*) echo "error: owners must match [A-Za-z0-9._-]+ (got '$GH_OWNER' / '$HF_OWNER')" >&2; exit 1;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

targets() {
  git -C "$ROOT" ls-files | while read -r f; do printf '%s/%s\n' "$ROOT" "$f"; done
  for extra in "$@"; do
    if [ -d "$extra" ]; then find "$extra" -type f \( -name '*.md' -o -name '*.cff' -o -name '*.py' -o -name '*.json' -o -name '*.txt' -o -name 'NOTICE*' -o -name 'LICENSE*' \); else echo "$extra"; fi
  done
}

targets "$@" | sort -u | while read -r f; do
  case "$f" in *.png|*.pdf|*.safetensors|*/set_identity.sh) continue;; esac
  if grep -q "__GH_OWNER__\|__HF_OWNER__" "$f" 2>/dev/null; then
    sed -i.bak -e "s/__GH_OWNER__/${GH_OWNER}/g" -e "s/__HF_OWNER__/${HF_OWNER}/g" "$f"
    rm -f "$f.bak"
    echo "set identity in: $f"
  fi
done

echo
echo "verification (should print nothing):"
grep -rn "__GH_OWNER__\|__HF_OWNER__" "$ROOT" --exclude-dir=.git --exclude='*.png' --exclude='*.pdf' --exclude='set_identity.sh' || echo "  clean — identity set to github.com/${GH_OWNER}, huggingface.co/${HF_OWNER}"
