---
name: uno-latest-checkpoint-infer
description: Download the newest checkpoint from the Hugging Face UNO GRPO repository, update only `setting` and `checkpoint_step` in `infer_benchmark_RL.py`, run a selected UNO benchmark on GPU 3, and calculate its paired metrics on that same GPU. Use when asked to pull a new/latest UNO checkpoint from `emiliiia/makeupGRPO_26_7_20`, evaluate BeautyBench or another UNO benchmark, calculate paired metrics, or repeat the UNO checkpoint-to-benchmark workflow.
---

# UNO latest checkpoint inference

Use this workflow for `/nfs_share/yuyue/cvpr_eval/results/UNO/UNO`.

1. Obtain a Hugging Face access token from the user if one is not already available. Never write it into a file, source code, command history, or final response. Supply it only through the `HF_TOKEN` environment variable for the command that needs it.
2. Run the bundled downloader from the repository root. It discovers the most recently committed `checkpoint-*` containing `diffusion_pytorch_model.safetensors`, downloads every file beneath that checkpoint while preserving its Hub path below `/nfs_share/yuyue/cvpr_eval/results/UNO/ckpt`, and verifies LFS SHA-256 values when supplied by Hub.

   ```bash
   HF_TOKEN="$HF_TOKEN" python /home/wangjiayu/.codex/skills/uno-latest-checkpoint-infer/scripts/fetch_latest_checkpoint.py
   ```

   The downloader creates or refreshes the matching symlink beneath `UNO/UNO/ckpt` so the existing inference code can resolve the checkpoint without changing its path template. It patches only `checkpoint_step` and `setting` in `infer_benchmark_RL.py`.
3. Inspect the downloader's reported setting, checkpoint step, destination, and checksum result. Stop on a failed download or checksum mismatch; do not run inference.
4. Before expensive inference, preflight the selected GPU and PyTorch/CUDA import. Stop and report the environment error if this fails; do not attempt inference or metrics.

   ```bash
   source /home/wangjiayu/miniconda3/bin/activate && conda activate uno_env && ENV_LIBS=$(find "$CONDA_PREFIX/lib/python3.10/site-packages/nvidia" -type d -name lib -print | paste -sd:) && export LD_LIBRARY_PATH="$ENV_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" && CUDA_VISIBLE_DEVICES=3 python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'
   ```

   Always set this conda-provided CUDA library path after activating either environment. It must precede the inherited `LD_LIBRARY_PATH`, otherwise the broken shared CUDA library may be loaded instead.

5. Normalize the requested benchmark before running it. Default to `BeautyBench` when no dataset is specified. Accept case-insensitive names, spaces, hyphens, and underscores; map `beautybench` to `BeautyBench`, `mt` to `MT`, `mtwild` to `MT_wild`, and `ladn` to `LADN`. Reject any other input rather than guessing. `infer_benchmark_RL.py` currently sets `DATA_LIST = None`, so always pass the normalized `--dataset`.

   ```bash
   source /home/wangjiayu/miniconda3/bin/activate && conda activate uno_env && ENV_LIBS=$(find "$CONDA_PREFIX/lib/python3.10/site-packages/nvidia" -type d -name lib -print | paste -sd:) && export LD_LIBRARY_PATH="$ENV_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" && export http_proxy=http://127.0.0.1:3316 && export https_proxy=http://127.0.0.1:3316 && export no_proxy="localhost,127.0.0.1" && cd /nfs_share/yuyue/cvpr_eval/results/UNO/UNO && CUDA_VISIBLE_DEVICES=3 python infer_benchmark_RL.py --dataset BeautyBench
   ```

6. Immediately calculate paired metrics for the same normalized dataset on the same physical GPU after its inference succeeds. Preflight the `cvpr_eval` environment with the analogous `import torch` command first. Use the `cvpr_eval` environment and restrict visibility to GPU 3; this makes all four metric models use its single visible device. The bundled wrapper reads the current `setting`, `checkpoint_step`, and `guidance`, writes per-image JSON plus an `_average.json`, and does not modify `metrics/models.yaml`. Pass `--guidance` explicitly when evaluating an older run after the source default has been changed.

   ```bash
   source /nfs_share/yuyue/miniconda3/bin/activate && conda activate cvpr_eval && ENV_LIBS=$(find "$CONDA_PREFIX/lib/python3.10/site-packages/nvidia" -type d -name lib -print | paste -sd:) && export LD_LIBRARY_PATH="$ENV_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" && cd /nfs_share/yuyue/cvpr_eval && CUDA_VISIBLE_DEVICES=3 PYTHONPATH="." python /home/wangjiayu/.codex/skills/uno-latest-checkpoint-infer/scripts/calculate_paired_metrics.py --dataset BeautyBench
   ```

   Do not run `metrics/average_paired_metrics.py` until results for the intended set of benchmarks exist. Do not run FID during the current BeautyBench-only stage.
7. Report the selected Hub checkpoint, normalized benchmark name, output directory, and paired-metrics average JSON. Do not delete an existing output directory without explicit authorization: the current inference script does delete the selected dataset subdirectory before generating it.

Use `infer_benchmark.py` only when the user explicitly asks for the SFT benchmark workflow; it does not use `setting` and `checkpoint_step`.
