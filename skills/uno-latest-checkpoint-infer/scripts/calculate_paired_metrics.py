#!/usr/bin/env python3
"""Calculate paired metrics for one current UNO RL benchmark result."""

import argparse
import re
from pathlib import Path

import metrics.calculate_paired_metrics as paired

PROJECT = Path("/nfs_share/yuyue/cvpr_eval/results/UNO/UNO")
INFER_SCRIPT = PROJECT / "infer_benchmark_RL.py"
MODEL_NAME = "UNO_RL_current"


def normalize_dataset(value: str) -> str:
    key = re.sub(r"[\s_-]+", "", value).lower()
    datasets = {
        "beautybench": "BeautyBench",
        "mt": "MT",
        "mtwild": "MT_wild",
        "ladn": "LADN",
    }
    if key not in datasets:
        allowed = ", ".join(datasets.values())
        raise argparse.ArgumentTypeError(f"Unsupported dataset '{value}'. Choose one of: {allowed}")
    return datasets[key]


def current_selectors():
    source = INFER_SCRIPT.read_text()
    setting = re.search(r'(?m)^setting\s*=\s*["\']([^"\']+)["\']\s*$', source)
    step = re.search(r"(?m)^checkpoint_step\s*=\s*(\d+)\s*$", source)
    guidance = re.search(r"(?m)^\s*guidance:\s*float\s*=\s*([0-9.]+)\s*$", source)
    if not setting or not step or not guidance:
        raise RuntimeError("Could not read setting, checkpoint_step, and guidance from infer_benchmark_RL.py")
    return setting.group(1), int(step.group(1)), float(guidance.group(1))


def format_guidance(value: float) -> str:
    return f"{value:g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=normalize_dataset, default="BeautyBench")
    parser.add_argument("--setting")
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--guidance", type=float)
    args = parser.parse_args()
    setting_in_file, step_in_file, guidance_in_file = current_selectors()
    setting = args.setting or setting_in_file
    step = args.checkpoint_step or step_in_file
    guidance = args.guidance if args.guidance is not None else guidance_in_file
    checkpoint = f"checkpoint-{step}-guidance{format_guidance(guidance)}"
    paired.MODELS_REGISTRY[MODEL_NAME] = {
        "result_path": f"results/UNO_RL/{setting}/{checkpoint}/{{dataset}}",
        "metric_result_path": f"results/UNO_RL/metric_results_{setting}/{checkpoint}/{{dataset}}.json",
        "img_ext": "png",
    }
    paired.evaluate_metrics(
        MODEL_NAME,
        args.dataset,
        {"background": 0, "clip": 0, "dino": 0, "face": 0},
    )


if __name__ == "__main__":
    main()
