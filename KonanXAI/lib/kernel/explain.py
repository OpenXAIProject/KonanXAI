from abc import *

from KonanXAI.lib.core.darknet.yolo import BBox
from ...utils import compose_heatmap_image
import cv2

class ExplainData:
    def __init__(self, results: dict, mtype: str, platform: str):
        self.explain = results
        self.mtype = mtype
        self.platform = platform
    @abstractmethod
    def __getitem__(self, ):
        pass
    
    def save_heatmap(self, save_path: str, ext: str=".jpg"):
        draw_box, bbox = False,None
        for algorithm, datasets in self.explain.items():
            for i, data in enumerate(datasets):
                results, origin_image = data
                if isinstance(results, tuple):
                    bbox_li = results[1]
                    results = results[0]
                for j, saliency in enumerate(results):
                    saliency_path = f"{save_path}{algorithm}_saliency_{i}-{j}{ext}"
                    compose_path = f"{save_path}{algorithm}_compose_{i}-{j}{ext}"
                    cv2.imwrite(saliency_path, saliency)   
                    if 'yolo' in self.mtype:
                        draw_box = True
                        t = bbox_li[j]
                        if self.platform == 'darknet':
                            bbox = BBox(t.in_w, t.in_h,t.cx,t.cy,t.w,t.h,t.entry,t.class_idx,t.class_probs,t.probs)
                            bbox = bbox.to_xyxy()
                        elif self.platform == 'pytorch':
                            bbox = list(map(int,t))
                            bbox = (bbox[0],bbox[1]), (bbox[2],bbox[3])
                    compose_heatmap_image(saliency, origin_image, bbox, save_path=compose_path, draw_box= draw_box)