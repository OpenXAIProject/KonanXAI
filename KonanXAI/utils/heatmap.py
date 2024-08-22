import numpy as np
import cv2

def compose_heatmap_image(saliency, origin_image, bbox=None, ratio=0.5, save_path=None, name=None, draw_box=False):
    origin_image = np.array(origin_image)
    origin_image = cv2.cvtColor(origin_image,cv2.COLOR_BGR2RGB)
    result = origin_image // 2 + saliency // 2
    result = result.astype(np.float32)
    min_value = np.min(result)
    max_value = np.max(result)
    result = (result - min_value) / (max_value - min_value) * 255
    result = result.astype(np.uint8)
    if draw_box:
        result = cv2.rectangle(result,bbox[0],bbox[1],color=(0,255,0),thickness=3)
    cv2.imwrite(save_path, result)