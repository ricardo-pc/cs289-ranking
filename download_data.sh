#!/usr/bin/env bash
# Downloads MovieLens 1M into data/raw/ml-1m/
# Run once from the repo root: bash download_data.sh

set -e

TARGET="data/raw"
ZIP="$TARGET/ml-1m.zip"
DIR="$TARGET/ml-1m"

if [ -d "$DIR" ]; then
  echo "data/raw/ml-1m/ already exists — skipping download."
  exit 0
fi

mkdir -p "$TARGET"
echo "Downloading MovieLens 1M (~6MB)..."
curl -L "https://files.grouplens.org/datasets/movielens/ml-1m.zip" -o "$ZIP"
unzip -q "$ZIP" -d "$TARGET"
rm "$ZIP"
echo "Done. Files are in $DIR/"
echo ""
echo "Expected files:"
ls "$DIR/"
