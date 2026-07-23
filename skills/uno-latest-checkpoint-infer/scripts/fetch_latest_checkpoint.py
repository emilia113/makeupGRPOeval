#!/usr/bin/env python3
"""Fetch the newest UNO checkpoint and update the two RL inference selectors."""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = "emiliiia/makeupGRPO_26_7_20"
DOWNLOAD_ROOT = Path("/nfs_share/yuyue/cvpr_eval/results/UNO/ckpt")
CODE_CKPT_ROOT = Path("/nfs_share/yuyue/cvpr_eval/results/UNO/UNO/ckpt")
PROJECT = Path("/nfs_share/yuyue/cvpr_eval/results/UNO/UNO")
INFER_SCRIPT = PROJECT / "infer_benchmark_RL.py"
MODEL_NAME = "diffusion_pytorch_model.safetensors"


def get_json(url: str, token: str):
    request = Request(url, headers={"Authorization": f"Bearer {token}"})
    with urlopen(request) as response:
        return json.load(response)


def newest_checkpoint(entries):
    candidates = []
    for item in entries:
        if item.get("type") != "file" or not item.get("path", "").endswith("/" + MODEL_NAME):
            continue
        parent = item["path"].rsplit("/", 1)[0]
        match = re.fullmatch(r"(.+)/(checkpoint-(\d+)(?:-(\d+))?)", parent)
        if not match:
            continue
        date = item.get("lastCommit", {}).get("date", "")
        try:
            timestamp = datetime.fromisoformat(date.replace("Z", "+00:00")).timestamp()
        except ValueError:
            timestamp = 0
        candidates.append((timestamp, int(match.group(3)), int(match.group(4) or 0), parent, match.group(1)))
    if not candidates:
        raise RuntimeError("No checkpoint containing diffusion_pytorch_model.safetensors was found.")
    return max(candidates)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(repo: str, item: dict, destination: Path, token: str):
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = item.get("lfs", {}).get("oid")
    if destination.exists() and expected and sha256(destination) == expected:
        print(f"Verified existing {destination}")
        return
    partial = destination.with_name(destination.name + ".partial")
    expected_size = item.get("lfs", {}).get("size") or item.get("size")
    if partial.exists() and expected_size and partial.stat().st_size == expected_size:
        if expected and sha256(partial) != expected:
            raise RuntimeError(f"SHA-256 mismatch for interrupted download {item['path']}")
        partial.replace(destination)
        print(f"Finalized verified download {destination}")
        return
    url = f"https://huggingface.co/{repo}/resolve/main/{quote(item['path'])}"
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Authorization": f"Bearer {token}"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    print(f"Downloading {item['path']} from byte {offset}", flush=True)
    with urlopen(request) as response:
        append = offset and response.status == 206
        if offset and not append:
            offset = 0
        with partial.open("ab" if append else "wb") as handle:
            while chunk := response.read(16 * 1024 * 1024):
                handle.write(chunk)
    if expected and sha256(partial) != expected:
        raise RuntimeError(f"SHA-256 mismatch for {item['path']}")
    partial.replace(destination)
    print(f"Downloaded and verified {destination}", flush=True)


def patch_selectors(setting: str, step: int):
    source = INFER_SCRIPT.read_text()
    updated, count_step = re.subn(r"(?m)^checkpoint_step\s*=\s*\d+\s*$", f"checkpoint_step = {step}", source, count=1)
    updated, count_setting = re.subn(r'(?m)^setting\s*=\s*["\'][^"\']*["\']\s*$', f'setting = "{setting}"', updated, count=1)
    if count_step != 1 or count_setting != 1:
        raise RuntimeError("Could not uniquely update checkpoint_step and setting in infer_benchmark_RL.py")
    INFER_SCRIPT.write_text(updated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=REPO)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN to a Hugging Face token before running this script.")
    entries = get_json(f"https://huggingface.co/api/models/{args.repo}/tree/main?recursive=true&expand=true", token)
    _, step, _, checkpoint_dir, setting = newest_checkpoint(entries)
    selected = [item for item in entries if item.get("type") == "file" and item.get("path", "").startswith(checkpoint_dir + "/")]
    for item in selected:
        download_file(args.repo, item, DOWNLOAD_ROOT / item["path"], token)
    target = DOWNLOAD_ROOT / setting
    link = CODE_CKPT_ROOT / setting
    if link.is_symlink() or not link.exists():
        link.unlink(missing_ok=True)
        link.symlink_to(target)
    elif link.resolve() != target.resolve():
        raise RuntimeError(f"Refusing to replace existing non-symlink checkpoint directory: {link}")
    patch_selectors(setting, step)
    print(f"Selected: {checkpoint_dir}")
    print(f"Inference setting: {setting}; checkpoint_step: {step}")
    print(f"Checkpoint root: {DOWNLOAD_ROOT / checkpoint_dir}")


if __name__ == "__main__":
    main()
