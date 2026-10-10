#!/usr/bin/env bash
# Fetch the six baseline repositories at the pinned commits recorded in third_party/code/<name>/SOURCE.json,
# then verify against third_party/code/MANIFEST.sha256 (the exact files used in our runs).
set -euo pipefail
cd "$(dirname "$0")/.."
for dir in third_party/code/*/; do
  name=$(basename "$dir"); src="$dir/SOURCE.json"; [ -f "$src" ] || continue
  url=$(python3 -c "import json;print(json.load(open('$src'))['url'])")
  commit=$(python3 -c "import json;print(json.load(open('$src'))['commit'])")
  tmp=$(mktemp -d); git clone --quiet "$url" "$tmp"; git -C "$tmp" checkout --quiet "$commit"
  rsync -a --exclude .git "$tmp"/ "$dir"/; rm -rf "$tmp"; echo "fetched $name @ $commit"
done
echo "verifying against third_party/code/MANIFEST.sha256 ..."
sha256sum --quiet -c third_party/code/MANIFEST.sha256 && echo "all files match the versions used in the experiments"
