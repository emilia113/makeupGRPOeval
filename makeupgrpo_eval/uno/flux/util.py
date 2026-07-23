# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright (c) 2024 Black Forest Labs and The XLabs-AI Team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass

import torch
import json
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file as load_sft

from .model import Flux, FluxParams
from .modules.autoencoder import AutoEncoder, AutoEncoderParams
from .modules.conditioner import HFEmbedder

import re
from uno.flux.modules.layers import DoubleStreamBlockLoraProcessor, SingleStreamBlockLoraProcessor
def load_model(ckpt, device='cpu'):
    if ckpt.endswith('safetensors'):
        from safetensors import safe_open
        pl_sd = {}
        with safe_open(ckpt, framework="pt", device=device) as f:
            for k in f.keys():
                pl_sd[k] = f.get_tensor(k)
    else:
        pl_sd = torch.load(ckpt, map_location=device)
    return pl_sd

def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def get_lora_rank(checkpoint):
    for k in checkpoint.keys():
        if k.endswith(".down.weight"):
            return checkpoint[k].shape[0]

def load_checkpoint(local_path, repo_id, name):
    if local_path is not None:
        if '.safetensors' in local_path:
            print(f"Loading .safetensors checkpoint from {local_path}")
            checkpoint = load_safetensors(local_path)
        else:
            print(f"Loading checkpoint from {local_path}")
            checkpoint = torch.load(local_path, map_location='cpu')
    elif repo_id is not None and name is not None:
        print(f"Loading checkpoint {name} from repo id {repo_id}")
        checkpoint = load_from_repo_id(repo_id, name)
    else:
        raise ValueError(
            "LOADING ERROR: you must specify local_path or repo_id with name in HF to download"
        )
    return checkpoint


def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None
    repo_id_ae: str | None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-fp8": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV_FP8"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))

def load_from_repo_id(repo_id, checkpoint_name):
    ckpt_path = hf_hub_download(repo_id, checkpoint_name)
    sd = load_sft(ckpt_path, device='cpu')
    return sd

def load_flow_model(name: str, device: str | torch.device = "cuda", hf_download: bool = True):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)
    
    with torch.device("meta" if ckpt_path is not None else device):
        model = Flux(configs[name].params).to(torch.bfloat16)

    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_model(ckpt_path, device=str(device))
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return model

def load_flow_model_only_lora(
    name: str,
    device: str | torch.device = "cuda",
    hf_download: bool = True,
    lora_rank: int = 16,
    use_fp8: bool = False
):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    # 1. 定义主模型Flux地址
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = os.environ.get("flux_ckpt", None) or hf_hub_download(configs[name].repo_id, configs[name].repo_flow.replace("sft", "safetensors"))   # 若本地没有且允许 hf_download，会把配置里的 repo_flow 文件名里的 sft 替换成 safetensors 后从 HuggingFace 拉取
    
    # 2. 下载官方lora
        lora_ckpt_path = os.environ.get("LORA", None)

    # 3. 使用configs创建Flux主模型
    with torch.device("meta" if ckpt_path is not None else device):
        model = Flux(configs[name].params)

    # 4. 调用 set_lora 挂载 LoRA 版注意力处理器. Lora注意力处理器包括了lora的权重, 所以相当于创建lora的layer同时将layer挂载到flux模型上
    model = set_lora(model, lora_rank, device="meta" if lora_ckpt_path is not None else device)

    if ckpt_path is not None:
        print("Loading lora")
        # 加载lora的权重字典. 
        # load_sft就是load_file， 读取 .safetensors 文件后返回一个字典，键是参数名，值是对应的 torch.Tensor
        lora_sd = load_sft(lora_ckpt_path, device=str(device)) if lora_ckpt_path.endswith("safetensors")\
            else torch.load(lora_ckpt_path, map_location='cpu')
        
        print("Loading main checkpoint")
        # load_sft doesn't support torch.device

        if ckpt_path.endswith('safetensors'):
            if use_fp8:
                print(
                    "####\n"
                    "We are in fp8 mode right now, since the fp8 checkpoint of XLabs-AI/flux-dev-fp8 seems broken\n"
                    "we convert the fp8 checkpoint on flight from bf16 checkpoint\n"
                    "If your storage is constrained"
                    "you can save the fp8 checkpoint and replace the bf16 checkpoint by yourself\n"
                )
                sd = load_sft(ckpt_path, device="cpu")
                sd = {k: v.to(dtype=torch.float8_e4m3fn, device=device) for k, v in sd.items()}
            else:
                # 加载flux主模型权重字典
                sd = load_sft(ckpt_path, device=str(device))
            # LoRA 权重字典里的参数加入主权重 sd形成完整的权重字典
            sd.update(lora_sd)
            # 把主模型和lora的权重都加载到当前的dit模型中
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        else:
            dit_state = torch.load(ckpt_path, map_location='cpu')
            sd = {}
            for k in dit_state.keys():
                sd[k.replace('module.','')] = dit_state[k]
            sd.update(lora_sd)
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
            model.to(str(device))
        print_load_warning(missing, unexpected)
    return model

# 摘自pipeline的load_ckpt函数, 实现加载我们的lora到model, 替换掉模型中的lora权重
def load_ckpt(model, ckpt_path, device="cpu"):
    if ckpt_path is not None:
        from safetensors.torch import load_file as load_sft
        if ckpt_path.endswith('safetensors'):
            # 加载输入ckpt位置上的权重为字典, keys是参数名, values是对应的torch.Tensor
            sd = load_sft(ckpt_path, device='cpu')
            # load_state_dict就是把ckpt中的权重按照参数名字加载到模型中
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        
        else:
            dit_state = torch.load(ckpt_path, map_location='cpu')
            sd = {}
            for k in dit_state.keys():
                sd[k.replace('module.','')] = dit_state[k]
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
            model.to(str(device))
            
        print(f"missing keys: {missing}\n\n\n\n\nunexpected keys: {unexpected}")

#  把LoRA 注意力处理器挂到 Flux 模型的注意力模块上
def set_lora(
    model: Flux,
    lora_rank: int,
    double_blocks_indices: list[int] | None = None,
    single_blocks_indices: list[int] | None = None,
    device: str | torch.device = "cpu",
) -> Flux:
    double_blocks_indices = list(range(model.params.depth)) if double_blocks_indices is None else double_blocks_indices
    single_blocks_indices = list(range(model.params.depth_single_blocks)) if single_blocks_indices is None \
                            else single_blocks_indices
    
    lora_attn_procs = {}
    # 创建Lora注意力处理器
    with torch.device(device):
        for name, attn_processor in  model.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))

            if name.startswith("double_blocks") and layer_index in double_blocks_indices:
                lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=model.params.hidden_size, rank=lora_rank)
            elif name.startswith("single_blocks") and layer_index in single_blocks_indices:
                lora_attn_procs[name] = SingleStreamBlockLoraProcessor(dim=model.params.hidden_size, rank=lora_rank)
            else:
                lora_attn_procs[name] = attn_processor
    # 把创建好的lora注意力处理器挂到模型上
    model.set_attn_processor(lora_attn_procs)
    return model


def load_flow_model_quintized(name: str, device: str | torch.device = "cuda", hf_download: bool = True):
    # Loading Flux
    from optimum.quanto import requantize
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)
    # json_path = hf_hub_download(configs[name].repo_id, 'flux_dev_quantization_map.json')


    model = Flux(configs[name].params).to(torch.bfloat16)

    print("Loading checkpoint")
    # load_sft doesn't support torch.device
    sd = load_sft(ckpt_path, device='cpu')
    sd = {k: v.to(dtype=torch.float8_e4m3fn, device=device) for k, v in sd.items()}
    model.load_state_dict(sd, assign=True)
    return model
    with open(json_path, "r") as f:
        quantization_map = json.load(f)
    print("Start a quantization process...")
    requantize(model, sd, quantization_map, device=device)
    print("Model is quantized!")
    return model

def load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    version = os.environ.get("T5", "xlabs-ai/xflux_text_encoders")
    return HFEmbedder(version, max_length=max_length, torch_dtype=torch.bfloat16).to(device)

def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    version = os.environ.get("CLIP", "openai/clip-vit-large-patch14")
    return HFEmbedder(version, max_length=77, torch_dtype=torch.bfloat16).to(device)


def load_ae(name: str, device: str | torch.device = "cuda", hf_download: bool = True) -> AutoEncoder:
    ckpt_path = configs[name].ae_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_ae is not None
        and hf_download
    ):
        ckpt_path = os.environ.get("ae", None)

    # Loading the autoencoder
    print("Init AE")
    with torch.device("meta" if ckpt_path is not None else device):
        ae = AutoEncoder(configs[name].ae_params)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=device)
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return ae