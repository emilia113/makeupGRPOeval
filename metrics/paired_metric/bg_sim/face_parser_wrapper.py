# face_parser_wrapper.py
# 最终权威版: 封装了从本地源码加载和运行人脸解析模型的所有逻辑。

import sys
import os
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# --- 动态地将本地源码目录添加到Python的搜索路径中 ---
# !! 这是您存放 face-parsing.PyTorch 源码的绝对路径 !!
REPO_PATH = os.environ.get(
    "MAKEUPGRPO_FACE_PARSING_DIR",
    os.path.join(os.path.dirname(__file__), "face-parsing.PyTorch"),
)

if not os.path.isdir(REPO_PATH):
    raise FileNotFoundError(f"Face parsing repository not found at: '{REPO_PATH}'")

if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

# 现在，我们可以安全地从这个本地目录中导入 BiSeNet 了
try:
    from model import BiSeNet
except ImportError as e:
    raise ImportError(f"无法从 '{REPO_PATH}' 导入 BiSeNet。请确保该目录下存在 'model.py' 文件。原始错误: {e}")


class FaceParser:
    def __init__(self, device='cuda'):
        """
        初始化人脸解析器。
        """
        self.device = device
        self.n_classes = 19
        
        # 使用官方 README 指定的模型路径
        ckpt_path = os.path.join(REPO_PATH, 'res', 'cp', '79999_iter.pth')

        self.net = BiSeNet(n_classes=self.n_classes)
        self.net.to(self.device)
        self._load_checkpoint(ckpt_path)
        self.net.eval()
        print(f'* FaceParser: loaded checkpoint from local source: {ckpt_path}')

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # 标签定义
        self.parts_map = {
            'background': 0, 'skin': 1, 'l_brow': 2, 'r_brow': 3, 'l_eye': 4,
            'r_eye': 5, 'eye_g': 6, 'l_ear': 7, 'r_ear': 8, 'ear_r': 9,
            'nose': 10, 'mouth': 11, 'u_lip': 12, 'l_lip': 13, 'neck': 14,
            'neck_l': 15, 'cloth': 16, 'hair': 17, 'hat': 18
        }
        self.indices_map = {v: k for k, v in self.parts_map.items()}

    def _load_checkpoint(self, ckpt_path):
        """加载模型权重。"""
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"模型文件未找到: '{ckpt_path}'\n"
                f"请确保您已经创建了 'res/cp' 目录，并将 '79999_iter.pth' 模型文件放入其中。\n"
                f"下载链接: https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812"
            )
        self.net.load_state_dict(torch.load(ckpt_path, map_location=self.device))

    def parse(self, image_bgr):
        """对单张 BGR 格式的图像进行人脸解析。"""
        original_h, original_w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_resized = image_pil.resize((512, 512), Image.BILINEAR)
        img_tensor = self.to_tensor(image_resized)
        img_tensor = torch.unsqueeze(img_tensor, 0).to(self.device)
        with torch.no_grad():
            out = self.net(img_tensor)[0]
        parsing_anno = out.squeeze(0).cpu().numpy().argmax(0)
        parsing_resized = cv2.resize(parsing_anno.astype(np.uint8), 
                                     (original_w, original_h), 
                                     interpolation=cv2.INTER_NEAREST)
        masks = {}
        unique_indices = np.unique(parsing_resized)
        for idx in unique_indices:
            part_name = self.indices_map.get(idx)
            if part_name:
                mask = (parsing_resized == idx).astype(np.uint8) * 255
                masks[part_name] = mask
        return masks
