#!/usr/bin/env bash
set -euo pipefail

# Creates one reproducible snapshot of the exact four benchmark directories.
# Usage: scripts/create_benchmark_archive.sh /path/to/original/data [output-dir]
SOURCE_DATA=${1:?Usage: $0 SOURCE_DATA_DIR [OUTPUT_DIR]}
OUTPUT_DIR=${2:-assets_release}
ARCHIVE_NAME=makeupGRPOeval-benchmarks-v1.tar.zst

for dataset in BeautyBench MT MT_wild LADN; do
  test -f "$SOURCE_DATA/$dataset/${dataset}_pairs.json" || {
    echo "Missing benchmark asset: $SOURCE_DATA/$dataset/${dataset}_pairs.json" >&2
    exit 1
  }
done
mkdir -p "$OUTPUT_DIR"
tar --zstd -cf "$OUTPUT_DIR/$ARCHIVE_NAME" -C "$SOURCE_DATA" BeautyBench MT MT_wild LADN
sha256sum "$OUTPUT_DIR/$ARCHIVE_NAME" > "$OUTPUT_DIR/$ARCHIVE_NAME.sha256"
tar -tf "$OUTPUT_DIR/$ARCHIVE_NAME" > "$OUTPUT_DIR/$ARCHIVE_NAME.contents.txt"
echo "Created $OUTPUT_DIR/$ARCHIVE_NAME"
