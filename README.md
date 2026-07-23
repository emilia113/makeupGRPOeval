# makeupGRPOeval

Portable evaluation package for the four MakeupGRPO benchmarks:

| Benchmark | Pair JSON | Generation + metric support |
| --- | --- | --- |
| BeautyBench | `BeautyBench_pairs.json` | yes |
| MT | `MT_pairs.json` | yes |
| MT_wild | `MT_wild_pairs.json` | yes |
| LADN | `LADN_pairs.json` | yes |

It contains only the UNO-RL inference path, the four paired metrics, the
BiSeNet background-mask asset, and the three precomputed empty-prompt text
conditioning tensors used by the original UNO-RL evaluation. It intentionally
excludes all baseline methods, prior generated results, FLUX base weights, and
RL checkpoints.

## Restore on a new server

The target server needs the existing UNO Python environment and a GPU. From a
fresh clone:

```bash
git clone https://github.com/emilia113/makeupGRPOeval.git
cd makeupGRPOeval
# Activate the pre-existing UNO environment first. On this setup, ensure its
# CUDA libraries take precedence over inherited system libraries.
ENV_LIBS=$(find "$CONDA_PREFIX/lib/python3.10/site-packages/nvidia" -type d -name lib -print | paste -sd:)
export LD_LIBRARY_PATH="$ENV_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
pip install -r requirements-eval.txt
python scripts/download_benchmarks.py \
  --repo-id emiliiia/makeupGRPOeval-benchmarks \
  --output-dir data
```

The repository already includes `assets/text_cond_tensor/{txt.pt,txt_ids.pt,vec.pt}`.
They are required by UNO-RL inference and are loaded automatically. To use an
external copy instead, set `MAKEUPGRPO_TEXT_COND_DIR` to the directory that
contains those same three files.

UNO inference also loads its text encoders from Hugging Face on first use:
`xlabs-ai/xflux_text_encoders` (T5) and `openai/clip-vit-large-patch14` (CLIP).
The metric evaluator additionally downloads the models listed in
[`assets/MODELS.md`](assets/MODELS.md). For an air-gapped server, first run one
connected evaluation with the same environment and copy its Hugging Face cache
(normally `~/.cache/huggingface`, or the directory configured by `HF_HOME`) to
the destination; set `HF_HOME` there before running inference or metrics. The
FLUX base weights and the RL checkpoint remain explicit paths in the run command
below and must be made available separately.

The canonical benchmark download page is:

`https://huggingface.co/datasets/emiliiia/makeupGRPOeval-benchmarks`

The archive is a fresh,
single snapshot of `BeautyBench`, `MT`, `MT_wild`, and `LADN`, including their
pair JSONs and all images; its `.sha256` file must match before extracting.

## One RL run, then its metrics

The result layout is deliberately stable. Choose one `run-name` for every
parameter setting and retain it for metric computation.

```bash
CUDA_VISIBLE_DEVICES=0 python infer_benchmark_rl.py \
  --dataset BeautyBench \
  --data-dir data \
  --output-root outputs \
  --run-name results_lr2e-6/checkpoint-320-guidance8 \
  --checkpoint-path /path/to/checkpoint-320-0/diffusion_pytorch_model.safetensors \
  --flux-dir /path/to/flux-dev \
  --guidance 8

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python calculate_paired_metrics.py \
  --dataset BeautyBench \
  --data-dir data \
  --result-dir outputs/results_lr2e-6/checkpoint-320-guidance8/BeautyBench \
  --metric-dir outputs/metric_results_lr2e-6/checkpoint-320-guidance8
```

Repeat with `MT`, `MT_wild`, or `LADN`. Generation refuses to overwrite an
existing dataset output; use `--overwrite` only when intentionally rerunning
that exact dataset.

Metric outputs are:

```text
outputs/
  results_xxx/checkpoint-xxx-guidanceY/<BENCH>/<pair>_generated.png
  metric_results_xxx/checkpoint-xxx-guidanceY/<BENCH>.json
  metric_results_xxx/checkpoint-xxx-guidanceY/<BENCH>_average.json
```

The four metrics are `bg_l2` (lower is better), `face_sim`, `clipi`, and
`dino` (higher is better). Missing generated files are recorded as `null`, so
the average JSON also makes incomplete runs visible.

## Publishing the benchmark snapshot

On the source server, make a new archive from the *current extracted assets*,
not from any older zip files:

```bash
scripts/create_benchmark_archive.sh /nfs_share/yuyue/cvpr_eval/data assets_release
python scripts/upload_benchmarks_to_hf.py \
  --repo-id emiliiia/makeupGRPOeval-benchmarks \
  --archive assets_release/makeupGRPOeval-benchmarks-v1.tar.zst
```

The upload command creates a public Hugging Face **dataset** repository by
default. Authentication follows the standard local `HF_TOKEN`/HF CLI login;
never place a token in this repository.

## Metric assets

See [assets/MODELS.md](assets/MODELS.md). The evaluator uses public Hugging
Face IDs rather than absolute paths from the old server, so the destination
machine can populate its own cache. For strict comparisons, preserve the
downloaded model revisions/cache used by the experiment.
