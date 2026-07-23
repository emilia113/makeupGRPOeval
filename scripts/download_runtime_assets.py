#!/usr/bin/env python3
"""Download the versioned runtime checkpoints from the benchmark HF dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="emiliiia/makeupGRPOeval-benchmarks")
    parser.add_argument("--output-dir", type=Path, default=Path("assets/runtime"))
    args = parser.parse_args()
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=["runtime_assets/**"],
        local_dir=args.output_dir,
    )
    print(f"Downloaded runtime assets to {args.output_dir / 'runtime_assets'}")


if __name__ == "__main__":
    main()
