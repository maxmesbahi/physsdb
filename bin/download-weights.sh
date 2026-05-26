#!/usr/bin/env bash
# Download the PhysSDB pre-trained weights bundle from a GitHub Release
# into ./runs/.  Designed to be safe to re-run.
#
#   ./bin/download-weights.sh                       # latest release
#   ./bin/download-weights.sh v0.1.0                # specific tag
#
# Override the source repo with:
#   PHYSSDB_REPO=otheruser/physsdb ./bin/download-weights.sh
set -euo pipefail

REPO="${PHYSSDB_REPO:-YOUR_GH_USER/physsdb}"
TAG="${1:-latest}"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/runs"

if [ "$TAG" = "latest" ]; then
  URL="https://github.com/${REPO}/releases/latest/download/physsdb-weights.tar.gz"
else
  URL="https://github.com/${REPO}/releases/download/${TAG}/physsdb-weights.tar.gz"
fi

mkdir -p "$DEST_DIR"
echo "→ downloading $URL"
curl -L --fail --retry 3 --retry-delay 3 \
     -o "$DEST_DIR/physsdb-weights.tar.gz" "$URL"
echo "→ extracting into $DEST_DIR"
tar -xzf "$DEST_DIR/physsdb-weights.tar.gz" -C "$DEST_DIR" --strip-components=1
rm "$DEST_DIR/physsdb-weights.tar.gz"
echo "→ done:"
find "$DEST_DIR" -name model_best.pt | sed 's/^/    /'
