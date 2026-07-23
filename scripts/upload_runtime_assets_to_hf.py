#!/usr/bin/env python3
"""Upload the exact local runtime assets used by MakeupGRPO evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def add_folder(api: HfApi, repo_id: str, source: Path, destination: str) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            relative = path.relative_to(source)
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"{destination}/{relative}",
                repo_id=repo_id,
                repo_type="dataset",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="emiliiia/makeupGRPOeval-benchmarks")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-root", type=Path, default=Path("/nfs_share/yuyue/cvpr_eval"))
    parser.add_argument("--clip-dir", type=Path, required=True)
    args = parser.parse_args()
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=False, exist_ok=True)
    source = args.source_root
    add_folder(api, args.repo_id, args.clip_dir, "runtime_assets/clip-vit-large-patch14")
    add_folder(api, args.repo_id, source / "metrics/paired_metric/dino_i/dino_ckpt", "runtime_assets/dino-vitb16")
    api.upload_file(
        path_or_fileobj=str(source / "metrics/paired_metric/face_sim/20180402-114759-vggface2.pt"),
        path_in_repo="runtime_assets/face_sim/20180402-114759-vggface2.pt",
        repo_id=args.repo_id, repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(source / "metrics/paired_metric/bg_sim/face-parsing.PyTorch/res/cp/79999_iter.pth"),
        path_in_repo="runtime_assets/bg_sim/79999_iter.pth",
        repo_id=args.repo_id, repo_type="dataset",
    )
    add_folder(api, args.repo_id, args.project_root / "assets/text_cond_tensor", "runtime_assets/text_cond_tensor")
    flux_dir = source / "results/UNO/UNO/ckpt/flux-dev"
    api.upload_file(
        path_or_fileobj=str(flux_dir / "LICENSE.md"),
        path_in_repo="runtime_assets/uno/flux-dev/LICENSE.md",
        repo_id=args.repo_id, repo_type="dataset",
    )
    for filename in ("flux1-dev.safetensors", "ae.safetensors"):
        api.upload_file(
            path_or_fileobj=str(flux_dir / filename),
            path_in_repo=f"runtime_assets/uno/flux-dev/{filename}",
            repo_id=args.repo_id, repo_type="dataset",
        )
    # Make a local, Transformers-compatible T5 directory by pairing the
    # original FLUX text_encoder_2 weights with its tokenizer files.
    add_folder(api, args.repo_id, flux_dir / "text_encoder_2", "runtime_assets/uno/t5")
    add_folder(api, args.repo_id, flux_dir / "tokenizer_2", "runtime_assets/uno/t5")
    rl_root = source / "results/UNO/UNO/ckpt/part_latest_SFT_data_v3_cf_reward+cf+2e-6+GAS6"
    api.upload_file(
        path_or_fileobj=str(rl_root / "args.json"),
        path_in_repo="runtime_assets/uno/rl/part_latest_SFT_data_v3_cf_reward+cf+2e-6+GAS6/args.json",
        repo_id=args.repo_id, repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(rl_root / "checkpoint-320-0/diffusion_pytorch_model.safetensors"),
        path_in_repo="runtime_assets/uno/rl/part_latest_SFT_data_v3_cf_reward+cf+2e-6+GAS6/checkpoint-320-0/diffusion_pytorch_model.safetensors",
        repo_id=args.repo_id, repo_type="dataset",
    )
    print(f"Uploaded runtime assets to https://huggingface.co/datasets/{args.repo_id}/tree/main/runtime_assets")


if __name__ == "__main__":
    main()
