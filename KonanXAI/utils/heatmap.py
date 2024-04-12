import numpy as np
import cv2

def compose_heatmap_image(saliency, origin_image, ratio=0.5, save_path=None, name=None):
    origin_image = np.array(origin_image)
    # saliency = cv2.cvtColor(saliency, cv2.COLOR_BGR2RGB)
    result = origin_image // 2 + saliency // 2
    result = result.astype(np.float32)
    min_value = np.min(result)
    max_value = np.max(result)
    result = (result - min_value) / (max_value - min_value) * 255
    result = result.astype(np.uint8)
    cv2.imwrite(save_path, result)