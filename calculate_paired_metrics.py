#!/usr/bin/env python3
"""Calculate all four paired metrics for one UNO-RL benchmark output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from metrics.paired_metric.bg_sim.l2 import BackgroundSimilairty
from metrics.paired_metric.clip_i.clip_i import CLIPImageSimilarity
from metrics.paired_metric.dino_i.dino_i import DINOSimilarity
from metrics.paired_metric.face_sim.face_sim import FaceSimilarity

DATASETS = ("BeautyBench", "MT", "MT_wild", "LADN")


def dataset_name(value: str) -> str:
    normal = value.replace("-", "_").replace(" ", "_").lower()
    names = {name.lower(): name for name in DATASETS}
    if normal not in names:
        raise argparse.ArgumentTypeError(f"dataset must be one of: {', '.join(DATASETS)}")
    return names[normal]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=dataset_name)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--result-dir", type=Path, required=True,
                        help="directory containing <id>_generated.png")
    parser.add_argument("--metric-dir", type=Path, required=True,
                        help="results_xxx/metric_xxx destination directory")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def main() -> None:
    args = parse_args()
    pairs_path = args.data_dir / args.dataset / f"{args.dataset}_pairs.json"
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Missing pairs file: {pairs_path}")
    with pairs_path.open() as handle:
        pairs = json.load(handle)

    bg = BackgroundSimilairty(device=args.device)
    clip = CLIPImageSimilarity(device=args.device)
    dino = DINOSimilarity(device=args.device)
    face = FaceSimilarity(device=args.device)
    values: dict[str, dict] = {}
    dataset_dir = args.data_dir / args.dataset
    for pair_id, pair in tqdm(pairs.items(), desc=f"Metrics {args.dataset}"):
        source = dataset_dir / Path(pair[0]).relative_to(args.dataset)
        reference = dataset_dir / Path(pair[1]).relative_to(args.dataset)
        generated = args.result_dir / f"{pair_id}_generated.png"
        row = {"bareface": str(source), "makeup_reference": str(reference), "generated": str(generated)}
        if source.exists() and reference.exists() and generated.exists():
            source_image, reference_image, generated_image = load(source), load(reference), load(generated)
            row.update({
                "bg_l2": bg(source_image, generated_image),
                "face_sim": face(source_image, generated_image),
                "clipi": clip(reference_image, generated_image),
                "dino": dino(reference_image, generated_image),
            })
        else:
            row.update({"bg_l2": None, "face_sim": None, "clipi": None, "dino": None})
        values[pair_id] = row

    args.metric_dir.mkdir(parents=True, exist_ok=True)
    per_image = args.metric_dir / f"{args.dataset}.json"
    with per_image.open("w") as handle:
        json.dump(values, handle, indent=2)
    averages = {}
    for key in ("bg_l2", "clipi", "dino", "face_sim"):
        scores = [row[key][0] for row in values.values() if row[key] is not None]
        averages[key] = sum(scores) / len(scores) if scores else None
    with (args.metric_dir / f"{args.dataset}_average.json").open("w") as handle:
        json.dump(averages, handle, indent=2)
    print(f"Saved {per_image} and {args.metric_dir / f'{args.dataset}_average.json'}")


if __name__ == "__main__":
    main()
