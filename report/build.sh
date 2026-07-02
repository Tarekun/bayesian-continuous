#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building main.tex -> main.pdf ..."
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "Done. Output: $SCRIPT_DIR/main.pdf"
