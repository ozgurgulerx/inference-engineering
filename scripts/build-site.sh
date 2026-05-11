#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE_DIR="$ROOT/_site"

rm -rf "$SITE_DIR"
mkdir -p "$SITE_DIR"

cp "$ROOT/index.html" "$SITE_DIR/index.html"
cp "$ROOT/styles.css" "$SITE_DIR/styles.css"
cp "$ROOT/script.js" "$SITE_DIR/script.js"
cp -R "$ROOT/assets" "$SITE_DIR/assets"

cd "$ROOT/book"
"$ROOT/book/scripts/quarto" render

mkdir -p "$SITE_DIR/book"
cp -R "$ROOT/book/_book/." "$SITE_DIR/book/"
