import cv2
import numpy as np
import PIL.Image
from PIL import Image
from typing import Union, List
import torch

from metrics.paired_metric.bg_sim.face_parser_wrapper import FaceParser


class BackgroundSimilairty:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._build_model()

    def _build_model(self):
        self.face_parser = FaceParser(self.device)

    def get_background(self, image: Union[np.ndarray, PIL.Image.Image]) -> np.ndarray:
        """
        提取图像的“背景”区域：使用人脸解析掩码把脸部区域抹掉，仅保留非脸部像素。
        返回 BGR ndarray（uint8）。
        """
        # 输入统一成 BGR uint8 ndarray
        if isinstance(image, PIL.Image.Image):
            # PIL 是 RGB，需要转到 BGR
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:  # BGRA -> BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # 获取人脸解析掩码（假设 parse 返回一个 {part_name: HxW uint8 mask} 的 dict）
        masks = self.face_parser.parse(image)
        H, W = image.shape[:2]
        zeros_mask = np.zeros((H, W), dtype=np.uint8)

        # 聚合“脸部相关”区域作为需要排除的区域
        full_face_mask = (
            masks.get('skin',     zeros_mask) |
            masks.get('l_brow',   zeros_mask) |
            masks.get('r_brow',   zeros_mask) |
            masks.get('l_eye',    zeros_mask) |
            masks.get('r_eye',    zeros_mask) |
            masks.get('eye_g',    zeros_mask) |
            masks.get('nose',     zeros_mask) |
            masks.get('mouth',    zeros_mask) |
            masks.get('u_lip',    zeros_mask) |
            masks.get('l_lip',    zeros_mask)
        )

        # 生成背景图：非脸部区域保留原像素，脸部区域置 0
        background = np.zeros_like(image, dtype=np.uint8)
        background[full_face_mask < 127] = image[full_face_mask < 127]

        return background

    @staticmethod
    def _resize_to_match(src: np.ndarray, target_shape: tuple) -> np.ndarray:
        """把 src 按 target_shape(H, W) 进行双线性/面积插值缩放。"""
        th, tw = target_shape[:2]
        sh, sw = src.shape[:2]
        if (sh, sw) == (th, tw):
            return src
        interp = cv2.INTER_AREA if th < sh or tw < sw else cv2.INTER_LINEAR
        return cv2.resize(src, (tw, th), interpolation=interp)

    @staticmethod
    def _l2_distance(a: np.ndarray, b: np.ndarray, normalize_per_pixel: bool = True) -> float:
        """
        计算两张图的 L2 距离。
        - 先转 float32 并缩放到 [0,1]，避免图像大小影响到量级。
        - 若 normalize_per_pixel=True，则除以 sqrt(N*C) 进行每像素归一化，便于不同分辨率间可比。
        """
        a = a.astype(np.float32) / 255.0
        b = b.astype(np.float32) / 255.0
        diff = a - b
        l2 = float(np.linalg.norm(diff.ravel(), ord=2))
        if normalize_per_pixel:
            denom = np.sqrt(diff.size)  # N*C 的平方根, 消去分辨率的影响
            if denom > 0:
                l2 /= denom
        return l2

    @torch.no_grad()
    def __call__(
        self,
        generated_images: Union[Image.Image, List[Image.Image]],
        makeup_reference_images: Union[Image.Image, List[Image.Image]],
        return_background: bool = False,
    ) -> List[float]:
        """
        返回一个与输入成对对应的分数列表：背景 L2 距离（越小越相似）。
        """
        if isinstance(generated_images, Image.Image):
            generated_images = [generated_images]
        if isinstance(makeup_reference_images, Image.Image):
            makeup_reference_images = [makeup_reference_images]

        if len(generated_images) != len(makeup_reference_images):
            raise ValueError(
                f"generated_images 与 makeup_reference_images 数量不一致："
                f"{len(generated_images)} vs {len(makeup_reference_images)}"
            )

        if return_background:
            backgrounds: List[np.ndarray] = []
        
        scores: List[float] = []
        for gen_img, ref_img in zip(generated_images, makeup_reference_images):
            gen_bg = self.get_background(gen_img)
            ref_bg = self.get_background(ref_img)

            # 尺寸对齐（以参考图为基准）
            gen_bg = self._resize_to_match(gen_bg, ref_bg.shape)
            
            if return_background:
                backgrounds.append(gen_bg)
                backgrounds.append(ref_bg)

            # 计算 L2 距离（可切换是否按像素归一化）
            l2_dist = self._l2_distance(gen_bg, ref_bg, normalize_per_pixel=True)

            # 如果你更想返回“相似度”（越大越相似），可以这样：
            # sim = 1.0 / (1.0 + l2_dist)
            # scores.append(sim)
            scores.append(l2_dist)

        if return_background:
            return scores, backgrounds
        else:
            return scores




def main():
    
    generated_image = "./assets/16_sm.jpg"
    bareface_reference_image = "./assets/bareface_ref.jpg"
    device = "cuda:0"
    model = BackgroundSimilairty(device=device)
    score, backgrounds = model(
        generated_images=Image.open(generated_image),
        makeup_reference_images=Image.open(makeup_reference_image),
        return_background=True,
    )
    cv2.imwrite(f"./assets/background_gen.jpg", np.hstack(backgrounds))
    print("Background Similarity Score:", score)

if __name__ == "__main__":
    main()