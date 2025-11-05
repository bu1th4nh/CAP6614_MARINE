#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data/vqa_v2"
COCO_DIR="$ROOT_DIR/data/coco"

mkdir -p "$DATA_DIR" "$COCO_DIR"

# COCO val images
cd "$COCO_DIR"
if [ ! -d "val2014" ]; then
  wget http://images.cocodataset.org/zips/val2014.zip
  unzip -q val2014.zip
fi

# VQA v2 — questions + annotations
cd "$DATA_DIR"
mkdir -p annotations questions

wget -c https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -O questions/v2_Questions_Val_mscoco.zip
unzip -o -q questions/v2_Questions_Val_mscoco.zip -d questions

wget -c https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -O annotations/v2_Annotations_Val_mscoco.zip
unzip -o -q annotations/v2_Annotations_Val_mscoco.zip -d annotations

echo "Downloaded VQA v2 val questions + annotations"

