# Model assets

## Bundled UNO text conditioning

`assets/text_cond_tensor/` contains `txt.pt`, `txt_ids.pt`, and `vec.pt`: the
precomputed empty-prompt conditioning tensors used by the original UNO-RL
benchmark. They are tracked in this repository and are the default at runtime.
Set `MAKEUPGRPO_TEXT_COND_DIR` to override their location; the directory must
contain all three files.

## Hugging Face downloads

The repository deliberately does not ship the large Hugging Face model
snapshots cached on the original evaluation server. The evaluator downloads the
public model IDs on first use and then reuses the local Hugging Face cache:

| Consumer | Model ID / asset | Asset policy |
| --- | --- | --- |
| UNO inference (T5) | `xlabs-ai/xflux_text_encoders` | automatic download |
| UNO inference (CLIP text encoder) | `openai/clip-vit-large-patch14` | automatic download |
| CLIP-I | `openai/clip-vit-large-patch14` | automatic download |
| DINO-I | `facebook/dino-vitb16` | automatic download |
| Face similarity | VGGFace2 InceptionResnetV1 | `facenet-pytorch` automatic download |
| Background L2 | BiSeNet `79999_iter.pth` | tracked in this repository (53 MB) |

For air-gapped machines, pre-populate the Hugging Face cache on a connected
machine and copy it to the destination server. Set `HF_HOME` to that copied
cache (or use its default location, `~/.cache/huggingface`) before running.
Do not use model files from a different revision when comparing numbers with
prior experiments.
