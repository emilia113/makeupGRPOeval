#!/usr/bin/env python3
"""Download and verify the public benchmark archive, then extract it into data/."""

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="USERNAME/makeupGRPOeval-benchmarks")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--filename", default="makeupGRPOeval-benchmarks-v1.tar.zst")
    args = parser.parse_args()
    archive = Path(hf_hub_download(args.repo_id, args.filename, repo_type="dataset"))
    checksum = Path(hf_hub_download(args.repo_id, args.filename + ".sha256", repo_type="dataset"))
    expected = checksum.read_text().split()[0]
    actual = sha256(archive)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch: expected {expected}, got {actual}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(args.output_dir)], check=True)
    for dataset in ("BeautyBench", "MT", "MT_wild", "LADN"):
        path = args.output_dir / dataset / f"{dataset}_pairs.json"
        if not path.is_file():
            raise RuntimeError(f"Archive extraction incomplete: missing {path}")
    print(f"Verified and extracted benchmark assets to {args.output_dir}")


if __name__ == "__main__":
    main()
