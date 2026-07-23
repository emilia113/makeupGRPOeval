# 使用clip image embedding计算两张图之间的相似度作为妆容一致性的评价指标
# 包装为CLIPSimilarity类, 内含image_similarity方法用于计算相似度
# __init__方法中加载CLIP模型和预处理函数, image_similarity方法接受两张PIL图像作为输入, 返回相似度分数
import os
from typing import Union, List
import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, CLIPModel
import cv2
import torch.nn.functional as F



class CLIPImageSimilarity(object):
    def __init__(self, device):

        self.model_name = os.environ.get("MAKEUPGRPO_CLIP_MODEL", "openai/clip-vit-large-patch14")
        self.device = device
        self.build_model()

    def build_model(self):
        self.image_processor = AutoProcessor.from_pretrained(self.model_name)
        self.clip_model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
    
    # clip的相似度计算实现
    @torch.no_grad()
    def _clip_similarity(self, image1, image2):
        if not isinstance(image1, list):
            image1 = [image1]
        
        if not isinstance(image2, list):
            image2 = [image2]
        
        
        processed_image1 = self.image_processor(images=image1, return_tensors="pt").to(self.device)
        processed_image2 = self.image_processor(images=image2, return_tensors="pt").to(self.device)
        
        image1_features = self.clip_model.get_image_features(**processed_image1).squeeze()
        image2_features = self.clip_model.get_image_features(**processed_image2).squeeze()
        

        cos_sim = F.cosine_similarity(image1_features, image2_features, dim=0)  # 沿着第dim维计算余眩相似度

        return float(cos_sim.float())

    @torch.no_grad()
    def __call__(
            self,
            generated_images: Union[Image.Image, List[Image.Image]],
            makeup_reference_images: Union[Image.Image, List[Image.Image]],
    ):
        
        if isinstance(generated_images, Image.Image):
            generated_images = [generated_images]
        if isinstance(makeup_reference_images, Image.Image):
            makeup_reference_images = [makeup_reference_images]
        
        makeup_sim = []
        for generated_img, makeup_ref_img in zip(generated_images, makeup_reference_images):
            
            makeup_sim_score = self._clip_similarity(generated_img, image2=makeup_ref_img)
            makeup_sim.append(makeup_sim_score)

        return makeup_sim  # 返回一个score列表
        


def main():
    
    generated_image = "./assets/16_sm.jpg"
    makeup_reference_image = "./assets/makeup_ref.jpg"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPImageSimilarity(device=device)
    score = model(
        generated_images=Image.open(generated_image),
        makeup_reference_images=Image.open(makeup_reference_image),
    )
    print("CLIP Image Similarity Score:", score)

if __name__ == "__main__":
    main()
