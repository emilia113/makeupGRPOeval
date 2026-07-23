# Model assets

## Bundled UNO text conditioning

`assets/text_cond_tensor/` contains `txt.pt`, `txt_ids.pt`, and `vec.pt`: the
precomputed empty-prompt conditioning tensors used by the original UNO-RL
benchmark. They are tracked in this repository and are the default at runtime.
Set `MAKEUPGRPO_TEXT_COND_DIR` to override their location; the directory must
contain all three files.

## Exact runtime assets on Hugging Face

`emiliiia/makeupGRPOeval-benchmarks/runtime_assets/` is a copy of the local
checkpoint files used by this project. Restore it with:

```bash
python scripts/download_runtime_assets.py --output-dir assets/runtime
```

| Consumer | Model ID / asset | Asset policy |
| --- | --- | --- |
| UNO inference (T5) | `runtime_assets/uno/t5/` | `export T5=.../runtime_assets/uno/t5` |
| UNO inference (CLIP text encoder) | `runtime_assets/clip-vit-large-patch14/` | `export CLIP=.../clip-vit-large-patch14` |
| CLIP-I | `runtime_assets/clip-vit-large-patch14/` | `export MAKEUPGRPO_CLIP_MODEL=.../clip-vit-large-patch14` |
| DINO-I | `runtime_assets/dino-vitb16/` | `export MAKEUPGRPO_DINO_MODEL=.../dino-vitb16` |
| Face similarity | `runtime_assets/face_sim/20180402-114759-vggface2.pt` | `export vggface2_path=...` |
| Background L2 | `runtime_assets/bg_sim/79999_iter.pth` | already in GitHub code package |
| FLUX base | `runtime_assets/uno/flux-dev/` | pass as `--flux-dir` |
| Current RL checkpoint | `runtime_assets/uno/rl/.../checkpoint-320-0/` | pass as `--checkpoint-path` |

For air-gapped machines, run the download script on a connected machine and
copy `assets/runtime/` to the target server. These paths use local files and do
not require a Hugging Face cache at runtime.
