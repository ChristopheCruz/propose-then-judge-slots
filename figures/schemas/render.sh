#!/usr/bin/env bash
# Rendu des schémas : PDF vectoriel dans docs/article/figures/ (article) et PNG ×2 ici (README), avec Chromium sans tête.
#   python3 figures/schemas/make_figures.py && bash figures/schemas/render.sh
set -euo pipefail
cd "$(dirname "$0")"
tmp=$(mktemp -d -p "$HOME")   # Chromium en snap ne lit pas /tmp
for svg in architecture protocol decomposition; do
  w=$(grep -o 'width="[0-9]*"' $svg.svg | head -1 | tr -dc 0-9)
  h=$(grep -o 'height="[0-9]*"' $svg.svg | head -1 | tr -dc 0-9)
  cp $svg.svg "$tmp/"
  printf '<html><head><style>@page{size:%spx %spx;margin:0}html,body{margin:0}img{display:block;width:%spx;height:%spx}</style></head><body><img src="%s.svg"></body></html>' \
    "$w" "$h" "$w" "$h" "$svg" > "$tmp/$svg.html"
  chromium --headless --no-sandbox --disable-gpu --no-pdf-header-footer --print-to-pdf="$tmp/$svg.pdf" "file://$tmp/$svg.html" 2>/dev/null
  chromium --headless --no-sandbox --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
    --window-size="$w,$h" --screenshot="$tmp/$svg.png" "file://$tmp/$svg.svg" 2>/dev/null
  mkdir -p ../../docs/article/figures
  cp "$tmp/$svg.pdf" ../../docs/article/figures/
  cp "$tmp/$svg.png" .
  echo "$svg : ${w}x${h}"
done
rm -rf "$tmp"
