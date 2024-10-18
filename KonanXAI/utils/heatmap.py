import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from darknet.yolo import BBox
__all__= ['get_guided_heatmap', 'get_heatmap', 'get_kernelshap_image', 'get_lime_image', 'get_scale_heatmap', 'get_box', 'get_ig_heatmap']
def deprocess_image(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

def normalize_heatmap(heatmap):
    heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
    heatmap = (heatmap - heatmap_min).div(heatmap_max-heatmap_min).data
    return heatmap

def heatmap_tensor(origin_img, heatmaps, img_size, algorithm_type, framework):
    
    for i, heatmap in enumerate(heatmaps):
        if 'cam' in algorithm_type.lower():
            heatmap = F.interpolate(heatmap, size = img_size, mode="bilinear", align_corners=False)
            heatmap = normalize_heatmap(heatmap)
            heatmap = cv2.applyColorMap(np.uint8(255*heatmap.squeeze().detach().cpu()),cv2.COLORMAP_JET)
            heatmap = torch.Tensor(heatmap).permute(2,0,1).unsqueeze(0)
            return heatmap
        
        elif 'lrp' in algorithm_type.lower():
            cmap = matplotlib.cm.bwr
            heatmap = heatmap / torch.max(heatmap)
            heatmap = (heatmap +1.)/2.
            rgb = cmap(heatmap.flatten())[...,0:3].reshape([heatmap.shape[-2], heatmap.shape[-1], 3])
            heatmap = np.uint8(rgb*255) 
            heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)
            heatmap = torch.Tensor(heatmap).permute(2,0,1).unsqueeze(0)
            return heatmap
    
def get_box(bbox_li, framework):
    bbox = []
    if framework == "darknet":
        boxes_li = []
        for t in bbox_li:
            box = BBox(t.in_w, t.in_h,t.cx,t.cy,t.w,t.h,t.entry,t.class_idx,t.class_probs,t.probs)
            box = box.to_xyxy()
            bbox.append(box)
    else:
        bbox_li = [list(map(int, box)) for box in bbox_li]
        for t in bbox_li:
            box = (t[0],t[1]), (t[2],t[3])
            bbox.append(box)
    return bbox

def get_guided_heatmap(heatmaps, img_save_path, img_size, algorithm_type, framework, metric, score):
    draw_box = False
    bbox = None
    if len(heatmaps)>2:
        heatmaps, bbox_li, guided_imgs = heatmaps
        bbox = get_box(bbox_li, framework)
        if len(bbox)!=0:
            draw_box = True
    else:
        heatmaps, guided_imgs = heatmaps
    print(f"Image saving.... save path: {img_save_path}")
    for i, (heatmap, guided_img) in enumerate(tqdm(zip(heatmaps, guided_imgs))):
        if metric == None:
            save_path = f"{img_save_path[:-4]}_{algorithm_type}_{i}.jpg"
            compose_save_path = save_path.replace(".jpg", "_compose.jpg")
        else:
            save_path = f"{img_save_path[:-4]}_{algorithm_type}_{i}_{metric}_{score}.jpg"
            compose_save_path = save_path.replace(".jpg", "_compose.jpg")
        heatmap = F.interpolate(heatmap, size = img_size, mode="bilinear", align_corners=False)
        heatmap = normalize_heatmap(heatmap)
        heatmap_mask = heatmap.squeeze(0).squeeze(0).cpu().numpy()
        heatmap_mask = cv2.merge([heatmap_mask, heatmap_mask, heatmap_mask])
        heatmap = cv2.applyColorMap(np.uint8(255*heatmap.squeeze().detach().cpu()),cv2.COLORMAP_JET)
        compose_guided_img = deprocess_image(heatmap_mask * guided_img).squeeze(0)
        cv2.imwrite(save_path, heatmap)
        if bbox != None:
            result = cv2.rectangle(compose_guided_img, bbox[i][0], bbox[i][1], color=(0,255,0),thickness=3)
            cv2.imwrite(compose_save_path, result)
        else:
            cv2.imwrite(compose_save_path, compose_guided_img)
            
def get_heatmap(origin_img, heatmaps, img_save_path, img_size, algorithm_type, framework,metric,score):
    draw_box = False
    bbox = None
    if len(heatmaps)>1:
        heatmaps, bbox_li = heatmaps
        bbox = get_box(bbox_li, framework)
        if len(bbox)!=0:
            draw_box = True
    print(f"Image saving.... save path: {img_save_path}")
    for i, heatmap in enumerate(tqdm(heatmaps)):
        if metric == None:
            save_path = f"{img_save_path[:-4]}_{algorithm_type}_{i}.jpg"
            compose_save_path = save_path.replace(".jpg", "_compose.jpg")
        else:
            save_path = f"{img_save_path[:-4]}_{algorithm_type}_{i}_{metric}_{score}.jpg"
            compose_save_path = save_path.replace(".jpg", "_compose.jpg")
        if 'cam' in algorithm_type:
            heatmap = F.interpolate(heatmap, size = img_size, mode="bilinear", align_corners=False)
            heatmap = normalize_heatmap(heatmap)
            heatmap = cv2.applyColorMap(np.uint8(255*heatmap.squeeze().detach().cpu()),cv2.COLORMAP_JET)
        elif 'lrp' in algorithm_type:
            cmap = matplotlib.cm.bwr
            heatmap = heatmap / torch.max(heatmap)
            heatmap = (heatmap +1.)/2.
            heatmap = heatmap.detach().cpu().numpy()
            rgb = cmap(heatmap.flatten())[...,0:3].reshape([heatmap.shape[-2], heatmap.shape[-1], 3])
            heatmap = np.uint8(rgb*255) 
            heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)
            if bbox != None:
                heatmap = cv2.rectangle(heatmap, bbox[i][0], bbox[i][1],color=(0,255,0),thickness=3)
        elif algorithm_type in ["gradient", "gradientxinput", "smoothgrad"]:
            heatmap = normalize_heatmap(heatmap)
            heatmap = np.array(heatmap.squeeze(0).cpu().detach()*255).transpose(1,2,0)
        else:
            heatmap = np.array(heatmap.squeeze(0).cpu().detach()*255).transpose(1,2,0)
        
        cv2.imwrite(save_path, heatmap)
        if bbox != None:
            compose_heatmap_image(heatmap, origin_img, bbox[i], save_path = compose_save_path, draw_box = draw_box, framework = framework)
        else:
            compose_heatmap_image(heatmap, origin_img, bbox, save_path = compose_save_path, draw_box = draw_box)

def compose_heatmap_image(saliency, origin_image, bbox=None, ratio=0.5, save_path=None, name=None, draw_box=False, framework=None):
    if framework != "darknet":
        origin_image = np.array(origin_image.squeeze(0).detach()*255).transpose(1,2,0)
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    result = origin_image // 2 + saliency // 2
    result = result.astype(np.uint8)
    if draw_box:
        result = cv2.rectangle(result, bbox[0], bbox[1], color=(0,255,0),thickness=3)
    cv2.imwrite(save_path, result)

def get_scale_heatmap(origin_img, heatmaps, img_save_path, img_size, algorithm_type, framework, metric, score):
    is_empty = True
    draw_box = False
    bbox = None
    if metric == None:
        save_path = f"{img_save_path[:-4]}_{algorithm_type}.jpg"
        compose_save_path = save_path.replace(".jpg", "_compose.jpg")
    else:
        save_path = f"{img_save_path[:-4]}_{algorithm_type}_{metric}_{score}.jpg"
        compose_save_path = save_path.replace(".jpg", "_compose.jpg")
    if len(heatmaps)>1:
        heatmaps, bbox_li = heatmaps
        bbox = get_box(bbox_li, framework)
        if len(bbox)!=0:
            draw_box = True
    print(f"Image saving.... save path: {img_save_path}")
    for index, sbox in enumerate(heatmaps):
        is_empty = False
        sbox = F.interpolate(sbox, size = img_size, mode="bilinear", align_corners=False)
        sbox = normalize_heatmap(sbox)
        if index == 0:
            heatmap = sbox
        else:
            heatmap = torch.where(heatmap > sbox, heatmap, sbox)
    if is_empty == False:
        heatmap = cv2.applyColorMap(np.uint8(255*heatmap.squeeze().detach().cpu()),cv2.COLORMAP_JET)
        cv2.imwrite(save_path, heatmap)
        ## compose
        if framework != "darknet":
            origin_img = np.array(origin_img.squeeze(0).detach()*255).transpose(1,2,0)
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        result = origin_img // 2 + heatmap // 2
        result = result.astype(np.uint8)
        if draw_box:
            for i in range(len(bbox)):
                result = cv2.rectangle(result, bbox[i][0], bbox[i][1], color=(0,255,0),thickness=3)
        cv2.imwrite(compose_save_path, result)
    else: 
        print("Check out the data set. There are no inferred values.")
        
def get_ig_heatmap(origin_img, heatmaps, img_save_path, img_size, algorithm_type, framework, metric, score):
    if framework != "darknet":
        origin_img = np.array(origin_img.squeeze(0).detach()*255).transpose(1,2,0)
    origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
    if not isinstance(heatmaps,(list,tuple,np.ndarray)):
        return
    if isinstance(heatmaps,list):
        for i, heatmap in enumerate(heatmaps):
            if metric == None:
                save_path = f"{img_save_path[:-4]}_{algorithm_type}_{i}.jpg"
                compose_save_path = save_path.replace(".jpg", "_compose.jpg")
            else:
                save_path = f"{img_save_path[:-4]}_{algorithm_type}_{i}_{metric}_{score}.jpg"
                compose_save_path = save_path.replace(".jpg", "_compose.jpg")
            ig_image, mixed_image = convert_ig_image(heatmap, origin_img)
            cv2.imwrite(save_path, ig_image)
            cv2.imwrite(compose_save_path, mixed_image)
    else:
        if metric == None:
            save_path = f"{img_save_path[:-4]}_{algorithm_type}.jpg"
            compose_save_path = save_path.replace(".jpg", "_compose.jpg")
        else:
            save_path = f"{img_save_path[:-4]}_{algorithm_type}_{metric}_{score}.jpg"
            compose_save_path = save_path.replace(".jpg", "_compose.jpg")
        ig_image, mixed_image = convert_ig_image(heatmaps, origin_img)
        cv2.imwrite(save_path, ig_image)
        cv2.imwrite(compose_save_path, mixed_image)
    
def get_lime_image(heatmap, img_save_path, metric, score):
    if metric == None:
        save_path = f"{img_save_path[:-4]}_LIME.jpg"
    else:
        save_path = f"{img_save_path[:-4]}_LIME_{metric}_{score}.jpg"
    heatmap = np.array(heatmap*255, dtype=np.uint8)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path,heatmap)
 
def get_kernelshap_image(origin_img, heatmap, img_savepath,framework,metric, score):
    if metric == None:
        save_path = f"{img_savepath[:-4]}_KernelSHAP.jpg"
    else:
        save_path = f"{img_savepath[:-4]}_KernelSHAP_{metric}_{score}.jpg"
    def rgba_to_rgb(rgba_image):
        background = np.ones((224, 224, 3), dtype=np.float32)
        alpha = rgba_image[:,:,3]
        for c in range(3):
            background[:,:,c] = rgba_image[:,:,c] * alpha + background[:,:,c] * (1 - alpha)
        return background
    colors = []
    for l in np.linspace(1,0,100):
        colors.append((245/255,39/255,87/255,l))
    for l in np.linspace(0,1,100):
        colors.append((24/255,196/255,93/255,l))
    cm = LinearSegmentedColormap.from_list("shap", colors)
    heatmap = rgba_to_rgb(cm(heatmap))
    
    if framework != "darknet":
        origin_img = np.array(origin_img.squeeze(0).detach()*255).transpose(1,2,0)
    origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
    heatmap = heatmap * 255
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    compose = heatmap * 0.5 + origin_img * 0.5
    cv2.imwrite(save_path, compose)
    
def convert_ig_image(heatmap, origin_img):
    positive = np.clip(heatmap, 0, 1)
    gray_ig = np.average(positive, axis=2)
    
    linear_attr = linear_transform(gray_ig, 99, 0, 0.0, plot_distribution=False)
    linear_attr = np.expand_dims(linear_attr, 2) * [0, 255, 0]

    alpha = 0.7
    origin = np.array(origin_img) 
    mixed_image = (1 - alpha) * origin + alpha * linear_attr.astype(np.uint8) 
    
    ig_image = np.array(linear_attr, dtype=np.uint8)
    mixed_image = np.array(mixed_image, dtype=np.uint8)
    return ig_image, mixed_image

def linear_transform(attributions, clip_above_percentile=99.9, clip_below_percentile=70.0, low=0.2, plot_distribution=False):
    m = compute_threshold_by_top_percentage(attributions, percentage=100-clip_above_percentile, plot_distribution=plot_distribution)
    e = compute_threshold_by_top_percentage(attributions, percentage=100-clip_below_percentile, plot_distribution=plot_distribution)
    transformed = (1 - low) * (np.abs(attributions) - e) / (m - e) + low
    transformed *= np.sign(attributions)
    transformed *= (transformed >= low)
    transformed = np.clip(transformed, 0.0, 1.0)
    return transformed

def compute_threshold_by_top_percentage(attributions, percentage=60, plot_distribution=True):
    if percentage < 0 or percentage > 100:
        raise ValueError('percentage must be in [0, 100]')
    if percentage == 100:
        return np.min(attributions)
    flat_attributions = attributions.flatten()
    attribution_sum = np.sum(flat_attributions)
    sorted_attributions = np.sort(np.abs(flat_attributions))[::-1]
    cum_sum = 100.0 * np.cumsum(sorted_attributions) / attribution_sum
    threshold_idx = np.where(cum_sum >= percentage)[0][0]
    threshold = sorted_attributions[threshold_idx]
    if plot_distribution:
        raise NotImplementedError 
    return threshold
