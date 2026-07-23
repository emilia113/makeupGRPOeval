import os
from typing import Union, List
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
import torch.nn.functional as F


class DINOSimilarity(object):
    def __init__(self, device: str = "cuda"):
        self.sim_model_name = os.environ.get("MAKEUPGRPO_DINO_MODEL", "facebook/dino-vitb16")
        self.device = device
        self._build_model()

    def _build_model(self):
        
        # 始化计算相似度的模型
        model_id = self.sim_model_name
        self.dino_processor = ViTImageProcessor.from_pretrained(model_id)
        self.dino_model = ViTModel.from_pretrained(model_id).to(self.device).eval()
    
    # dino的相似度计算实现
    @torch.no_grad()
    def _dino_similarity(self, image_list):
        inputs = self.dino_processor(images=image_list, return_tensors="pt").to(self.dino_model.device)
        image_embeddings = self.dino_model(**inputs).last_hidden_state[:, 0, :]     # (768, ) 
        # dino vit的last_hidden_state是transformer encoder最后一层输出的[CLS]token，也是能够表示整张图像信息embedding，即dino embedding
        # self.pooler是留给用户用于训练分类器的线性映射层，在diffuser官方给出的ckpt是没有这一项的初始化的
        # 应该取last hidden state这个图像序列的token 0
        cos_sim = F.cosine_similarity(image_embeddings[0], image_embeddings[1], dim=0)
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
            
            makeup_sim_score = self._dino_similarity(image_list=[generated_img, makeup_ref_img])
            makeup_sim.append(makeup_sim_score)

        return makeup_sim  # 返回一个score列表
        
    


def main():
    
    generated_image = "./assets/16_sm.jpg"
    makeup_reference_image = "./assets/makeup_ref.jpg"
    device = "cuda:0"
    model = DINOSimilarity(device=device)
    score= model(
        generated_images=Image.open(generated_image),
        makeup_reference_images=Image.open(makeup_reference_image),
    )
    print("DINO Similarity Score:", score)

if __name__ == "__main__":
    main()
