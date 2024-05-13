from abc import *

from KonanXAI.lib.core.darknet.yolo import BBox
from ...utils import compose_heatmap_image
import cv2

class ExplainData:
    def __init__(self, results: dict):
        self.explain = results

    @abstractmethod
    def __getitem__(self, ):
        pass
    
    def save_heatmap(self, save_path: str, ext: str=".jpg"):
        for algorithm, datasets in self.explain.items():
            for i, data in enumerate(datasets):
                results, origin_image = data
                for j, saliency in enumerate(results[0]):
                    saliency_path = f"{save_path}{algorithm}_saliency_{i}-{j}{ext}"
                    compose_path = f"{save_path}{algorithm}_compose_{i}-{j}{ext}"
                    cv2.imwrite(saliency_path, saliency)        
                    t = results[1][j]            
                    bbox = BBox(t.in_w, t.in_h,t.cx,t.cy,t.w,t.h,t.entry,t.class_idx,t.class_probs,t.probs)
                    print(bbox.class_idx)
                    bbox = bbox.to_xyxy()
                    compose_heatmap_image(saliency, origin_image,bbox, save_path=compose_path)