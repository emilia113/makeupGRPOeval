#!/usr/bin/env python3
"""Generate one of the four MakeupGRPO benchmark datasets with an UNO-RL LoRA.

The output contract is fixed: ``<output-root>/<run-name>/<dataset>/`` contains
``<pair-id>_generated.png``, ``_source.png`` and ``_reference.png``.  The
paired evaluator consumes the generated file names directly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "makeupgrpo_eval"))
from uno.flux.pipeline import UNOPipeline, preprocess_ref  # noqa: E402

DATASETS = ("BeautyBench", "MT", "MT_wild", "LADN")


def dataset_name(value: str) -> str:
    normal = value.replace("-", "_").replace(" ", "_").lower()
    names = {name.lower(): name for name in DATASETS}
    if normal not in names:
        raise argparse.ArgumentTypeError(f"dataset must be one of: {', '.join(DATASETS)}")
    return names[normal]


def non_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}; pass --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=dataset_name)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--run-name", required=True,
                        help="e.g. results_20260722/checkpoint-320-guidance8")
    parser.add_argument("--checkpoint-path", required=True, type=Path,
                        help="RL LoRA checkpoint (.safetensors or .pt)")
    parser.add_argument("--flux-dir", required=True, type=Path,
                        help="directory containing flux1-dev.safetensors and ae.safetensors")
    parser.add_argument("--guidance", type=float, default=8.0)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    pair_file = data_dir / args.dataset / f"{args.dataset}_pairs.json"
    if not pair_file.is_file():
        raise FileNotFoundError(f"Missing pairs file: {pair_file}")
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing RL checkpoint: {args.checkpoint_path}")
    for name in ("flux1-dev.safetensors", "ae.safetensors"):
        if not (args.flux_dir / name).is_file():
            raise FileNotFoundError(f"Missing {name} in --flux-dir {args.flux_dir}")

    os.environ["ae"] = str(args.flux_dir / "ae.safetensors")
    os.environ["flux_ckpt"] = str(args.flux_dir / "flux1-dev.safetensors")
    os.environ["LORA"] = str(args.checkpoint_path)
    output_dir = args.output_root / args.run_name / args.dataset
    non_empty_dir(output_dir, args.overwrite)

    with pair_file.open() as handle:
        pairs = json.load(handle)
    accelerator = Accelerator()
    pipeline = UNOPipeline("flux-dev", "cuda", False, only_lora=True, lora_rank=args.lora_rank)
    pipeline.load_ckpt(str(args.checkpoint_path))
    pipeline.model.to(accelerator.device)

    for pair_id, pair in tqdm(pairs.items(), desc=f"Generating {args.dataset}"):
        if len(pair) < 2:
            raise ValueError(f"Pair {pair_id} has fewer than two references: {pair}")
        refs = [Image.open(data_dir / Path(item).relative_to(f"{args.dataset}")) for item in pair]
        refs = [preprocess_ref(image, 512) for image in refs]
        image = pipeline(prompt="", width=args.width, height=args.height,
                         guidance=args.guidance, num_steps=args.num_steps,
                         seed=args.seed, ref_imgs=refs, pe="d")[0]
        image.save(output_dir / f"{pair_id}_generated.png")
        refs[0].save(output_dir / f"{pair_id}_source.png")
        refs[1].save(output_dir / f"{pair_id}_reference.png")


if __name__ == "__main__":
    main()
