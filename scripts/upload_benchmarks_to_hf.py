#!/usr/bin/env python3
"""Create/update the public Hugging Face dataset repository with benchmark assets."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. USERNAME/makeupGRPOeval-benchmarks")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--private", action="store_true", help="Public is the default.")
    args = parser.parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    files = [args.archive, args.archive.with_suffix(args.archive.suffix + ".sha256"),
             args.archive.with_suffix(args.archive.suffix + ".contents.txt")]
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(file)
        api.upload_file(path_or_fileobj=str(file), path_in_repo=file.name,
                        repo_id=args.repo_id, repo_type="dataset")
    print(f"https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
