import random
import numpy as np
import torch
from KonanXAI.utils.heatmap import *      
def save_image(model_name, algorithm_type, origin_img, heatmap, img_save_path, img_size, framework, metric=None, score=None):
    if "eigencam" in algorithm_type:
        get_scale_heatmap(origin_img, heatmap, img_save_path, img_size,algorithm_type, framework, metric, score)
    elif "guided" in algorithm_type:
        get_guided_heatmap(heatmap, img_save_path, img_size,algorithm_type, framework, metric, score)
    elif "ig" == algorithm_type:
        get_ig_heatmap(origin_img, heatmap, img_save_path, img_size,algorithm_type, framework, metric, score)
    elif "lime" == algorithm_type:
        get_lime_image(heatmap, img_save_path, metric, score)
    elif "kernelshap" == algorithm_type:
        get_kernelshap_image(origin_img, heatmap, img_save_path, framework, metric, score)
    else:
        get_heatmap(origin_img, heatmap, img_save_path, img_size,algorithm_type, framework, metric, score)

def set_seed(seed_value=777):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) 