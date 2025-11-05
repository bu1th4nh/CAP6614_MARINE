#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/Data/gqa"
mkdir -p "$DATA_DIR"/{images,questions,scene_graphs}

echo "[GQA] Downloading images (Stanford NLP mirror)…"
cd "$DATA_DIR/images"
wget -c https://nlp.stanford.edu/data/gqa/images.zip -O images.zip
unzip -q -o images.zip
# images/ contains *.jpg under subfolders already; if it extracted to images/images, fix path:
if [ -d "images" ] && [ ! -f "images.zip" ]; then
  shopt -s dotglob nullglob
  mv images/* . || true
  rmdir images || true
fi

echo "[GQA] Downloading questions v1.2 (Stanford NLP mirror)…"
cd "$DATA_DIR/questions"
wget -c https://nlp.stanford.edu/data/gqa/questions1.2.zip -O questions1.2.zip
unzip -q -o questions1.2.zip

echo "[GQA] (Optional) Downloading scene graphs…"
cd "$DATA_DIR/scene_graphs"
wget -c https://nlp.stanford.edu/data/gqa/sceneGraphs.zip -O sceneGraphs.zip || true
unzip -q -o sceneGraphs.zip || true

echo "[GQA] Done. Images:$DATA_DIR/images  Questions:$DATA_DIR/questions"

