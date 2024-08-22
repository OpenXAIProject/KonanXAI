import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
def normalize_heatmap(heatmap):
    heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
    heatmap = (heatmap - heatmap_min).div(heatmap_max-heatmap_min).data
    return heatmap

def get_heatmap(origin_img, heatmaps, img_save_path, img_size, algorithm_type):
    draw_box = False
    bbox_li = None
    if isinstance(heatmaps,(tuple,list)):
        heatmaps, bbox_li = heatmaps
        bbox_li = [list(map(int, box)) for box in bbox_li]
        draw_box = True
    for i, heatmap in enumerate(heatmaps):
        compose_save_path = img_save_path[:-4] + '_gradcam_compose_{}.jpg'.format(i)
        if 'cam' in algorithm_type.lower():
            heatmap = F.interpolate(heatmap, size = img_size, mode="bilinear", align_corners=False)
            heatmap = normalize_heatmap(heatmap)
            heatmap = cv2.applyColorMap(np.uint8(255*heatmap.squeeze().detach().cpu()),cv2.COLORMAP_JET)
        elif 'lrp' in algorithm_type.lower():
            cmap = matplotlib.cm.bwr
            heatmap = heatmap / torch.max(heatmap)
            heatmap = (heatmap +1.)/2.
            rgb = cmap(heatmap.flatten())[...,0:3].reshape([heatmap.shape[-2], heatmap.shape[-1], 3])
            heatmap = np.uint8(rgb*255) 
            heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)
            heatmap = cv2.rectangle(heatmap,(bbox_li[i][0],bbox_li[i][1]),(bbox_li[i][2],bbox_li[i][3]),color=(0,255,0),thickness=3)
        cv2.imwrite(img_save_path[:-4] + '_gradcam_{}.jpg'.format(i), heatmap)
        compose_heatmap_image(heatmap, origin_img, bbox_li[i], save_path = compose_save_path, draw_box = draw_box)

def compose_heatmap_image(saliency, origin_image, bbox=None, ratio=0.5, save_path=None, name=None, draw_box=False):
    origin_image = np.array(origin_image.squeeze(0).detach()*255).transpose(1,2,0)
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    result = origin_image // 2 + saliency // 2
    result = result.astype(np.uint8)
    if draw_box:
        result = cv2.rectangle(result,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color=(0,255,0),thickness=3)
    cv2.imwrite(save_path, result)