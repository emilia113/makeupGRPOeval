# from typing import List, Union

import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Union
from facenet_pytorch import MTCNN, InceptionResnetV1
import os

class FaceSimilarity(object):
    def __init__(self, device: str = "cuda"):
        
        self.device = device
        self._build_model()

    def _build_model(self):
        # facenet-pytorch resolves/downloads this public weight into its cache.
        # A caller may still set vggface2_path before invoking this script to use
        # an offline cache.
        self.mtcnn = MTCNN(     # 参数与facenet-pytorch的infer.ipynb一致
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                device=self.device
            )
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        print("FaceSimilarity model built successfully")


    @torch.no_grad()
    def __call__(
        self,
        generated_images: Union[Image.Image, List[Image.Image]],        # 传入图像
        reference_images: Union[Image.Image, List[Image.Image]],
    ) -> List[float]:
        
        if isinstance(generated_images, Image.Image):
            generated_images = [generated_images]
        if isinstance(reference_images, Image.Image):
            reference_images = [reference_images]
        
        
        face_similarities = []
        for generated_image, reference_image in zip(generated_images, reference_images):
            # mtcnn检测人脸并裁剪
            generated_face = self.mtcnn(generated_image)    # mtcnn输入是单个PIL 
            reference_face = self.mtcnn(reference_image)

            # Correct None checks without tensor truth-value ambiguity
            if (generated_face is None) or (reference_face is None):
                face_similarities.append(0.0)
                continue

            # Ensure shape is (1, 3, 160, 160)
            if isinstance(generated_face, torch.Tensor) and generated_face.ndim == 3:
                generated_face = generated_face.unsqueeze(0)
            if isinstance(reference_face, torch.Tensor) and reference_face.ndim == 3:
                reference_face = reference_face.unsqueeze(0)

            generated_face = generated_face.to(self.device, non_blocking=True)
            reference_face = reference_face.to(self.device, non_blocking=True)

            # 计算embedding
            gen_emb = self.resnet(generated_face).squeeze(0)
            ref_emb = self.resnet(reference_face).squeeze(0)
            
            # 计算相似度
            similarity = F.cosine_similarity(gen_emb, ref_emb, dim=0)       # (512, )和(512, )张量计算余弦相似度, 沿着dim=0计算
            face_similarities.append(float(similarity.item()))
                
        return face_similarities  # 返回一个score列表
    

def main():
    
    generated_image = "./assets/16_sm.jpg"
    bareface_reference_image = "./assets/bareface_ref.jpg"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FaceSimilarity(device=device)
    score = model(
        generated_images=Image.open(generated_image),
        reference_images=Image.open(bareface_reference_image),
    )
    print("Face Similarity Score:", score)

if __name__ == "__main__":
    main()
