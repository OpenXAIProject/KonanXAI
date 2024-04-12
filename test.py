"""import KonanXAI as XAI
from KonanXAI.lib.core import darknet
mtype = XAI.ModelType.Yolov4Tiny
# platform
platform = XAI.PlatformType.Darknet
# dataset
dtype = XAI.DatasetType.COCO
# explain
etype = XAI.ExplainType.GradCAM
xai = XAI.XAI()

xai.load_model_support(mtype, platform, pretrained=True)

net: darknet.Network = xai.model.net
img = darknet.open_image(r"test.jpg", (416, 416))

net.forward_image(img)
net.backward()
print(net)
pass"""

import KonanXAI as XAI
from KonanXAI.lib.core import darknet
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from KonanXAI.lib.algorithm import *


mtype = XAI.ModelType.Yolov4Tiny
# platform
platform = XAI.PlatformType.Darknet
# dataset
dtype = XAI.DatasetType.COCO
# explain
etype = XAI.ExplainType.GradCAMpp

xai = XAI.XAI()
xai.load_model_support(mtype, platform, pretrained=True)
# xai.load_dataset_support(dtype, maxlen=10, shuffle=False)
xai.load_dataset_support(dtype)
xai.set_explain_mode([XAI.ExplainType.GradCAMpp])
explain = xai.explain()
explain.save_heatmap("heatmap/")
pass
"""

# img = darknet.open_image(r"test.jpg", (416, 416))
# net: darknet.Network = xai.model.net

# # 0. Load Drawing Image
# dimg = cv2.imread(r"test.jpg")
# dimg = cv2.cvtColor(dimg, cv2.COLOR_BGR2RGB)
# rimg = cv2.resize(dimg, (416, 416), interpolation=cv2.INTER_LINEAR)
# dimg = cv2.resize(dimg, (416, 416), interpolation=cv2.INTER_LINEAR)

# # 1. Forward
# net.forward_image(img)

# # 2. get BBox
# bboxes = []
# bbox_target_layer = {}
# for i, layer in enumerate(net.layers):
#     if layer.type == darknet.LAYER_TYPE.YOLO:
#         boxes = layer.get_bboxes(threshold=0.9)
#         for bbox in boxes:
#             bbox_target_layer[bbox.entry] = layer
#         bboxes += boxes

# # 3. NMS
# bboxes = darknet.non_maximum_suppression_bboxes(bboxes)

# # 4. Drawing BBox
# for box in bboxes:
#     p1, p2 = box.to_xyxy()
#     cv2.rectangle(rimg, p1, p2, color=(0,0,255),thickness=1)

# cv2.imshow('Image', rimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # DEfine 
# def show_cam_on_image(img: np.ndarray,
#                       mask: np.ndarray,
#                       use_rgb: bool = False,
#                       colormap: int = cv2.COLORMAP_JET,
#                       image_weight: float = 0.5) -> np.ndarray:
#     This function overlays the cam mask on the image as an heatmap.
#     By default the heatmap is in BGR format.

#     :param img: The base image in RGB or BGR format.
#     :param mask: The cam mask.
#     :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
#     :param colormap: The OpenCV colormap to be used.
#     :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
#     :returns: The default image with the cam overlay.
#    
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask).squeeze(), colormap)
#     if use_rgb:
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     heatmap = np.float32(heatmap) / 255
#     if np.max(img) > 1:
#         raise Exception(
#             "The input image should np.float32 in the range [0, 1]")

#     if image_weight < 0 or image_weight > 1:
#         raise Exception(
#             f"image_weight should be in the range [0, 1].\
#                 Got: {image_weight}")

#     cam = (1 - image_weight) * heatmap + image_weight * img
#     cam = cam / np.max(cam)
#     return np.uint8(255 * cam)
# def scale_cam_image(cam, target_size=None):
#     result = []
#     for img in cam:
#         img = img - np.min(img)
#         img = img / (1e-7 + np.max(img))
#         if target_size is not None:
#             img = cv2.resize(img, target_size)
#         result.append(img)
#     result = np.float32(result)

#     return result

# def heatmap_normalize(saliency):
#     saliency_min, saliency_max = saliency.min(), saliency.max()
#     saliency = (saliency - saliency_min).div(saliency_max-saliency_min).data
#     return saliency

# def compose_heatmap_image(saliency, origin_image, ratio=0.5, save_path=None, name=None):
#     origin_image = np.array(origin_image)
#     result = origin_image // 2 + saliency // 2
#     result = result.astype(np.float32)
#     min_value = np.min(result)
#     max_value = np.max(result)
#     result = (result - min_value) / (max_value - min_value) * 255
#     result = result.astype(np.uint8)
#     cv2.imwrite(save_path, result)



# yolo_layer = [net.layers[30], net.layers[37]]
# # 5. BBox Backward
# for box in bboxes:
#     entry = box.entry
#     target_layer: darknet.Layer = bbox_target_layer[entry]
#     net.zero_grad()
#     stride = target_layer.out_w * target_layer.out_h
#     out = target_layer.get_output()
#     idx = entry + (5 + box.class_idx) * stride
#     target_layer.delta[idx] = out[idx]
#     # for i in range(target_layer.classes + 5):
#     #     idx = entry + i * stride
#     #     target_layer.delta[idx] = out[idx]
#     net.backward()
#     # prev_idx = layer.idx - 1
#     # prev_layer = net.layers[prev_idx]
#     # class_grad = np.array(prev_layer.get_delta()).reshape((-1, stride)).mean(1)
#     # feature_map = np.array(prev_layer.get_output()).reshape((-1, stride))
#     # weights = class_grad.reshape((-1, 1))
#     # mul = weights * feature_map
#     # saliency_map = np.sum(mul, axis=0).reshape((1, 1, target_layer.out_w, target_layer.out_h))
#     # # ReLU
#     # # saliency_map = np.where(saliency_map > 0, saliency_map, 0.0)
#     # saliency_map = F.relu(torch.tensor(saliency_map))

#     # resize = (416, 416)
#     # sailency = heatmap_normalize(saliency_map)
#     # heatmap = cv2.applyColorMap(np.uint8(255*sailency.squeeze().detach().cpu()), cv2.COLORMAP_JET)
#     # heatmap = cv2.resize(heatmap, dsize = resize, interpolation=cv2.INTER_LINEAR)
#     maps = []
#     for layer in yolo_layer:
#         prev_idx = layer.idx - 1
#         prev_layer = net.layers[prev_idx]
#         stride = prev_layer.out_w * prev_layer.out_h
#         class_grad = np.array(prev_layer.get_delta()).reshape((-1, stride)).mean(1)
#         feature_map = np.array(prev_layer.get_output()).reshape((-1, stride))
#         weights = class_grad.reshape((-1, 1))
#         mul = weights * feature_map
#         saliency_map = np.sum(mul, axis=0).reshape((1, 1, prev_layer.out_w, prev_layer.out_h))
#         # ReLU
#         # saliency_map = np.where(saliency_map > 0, saliency_map, 0.0)
#         saliency_map = F.relu(torch.tensor(saliency_map))

#         resize = (416, 416)
#         sailency = heatmap_normalize(saliency_map)
#         heatmap = cv2.applyColorMap(np.uint8(255*sailency.squeeze().detach().cpu()), cv2.COLORMAP_JET)
#         heatmap = cv2.resize(heatmap, dsize = resize, interpolation=cv2.INTER_LINEAR)
#         maps.append(heatmap)
#     sum_map = maps[0] + maps[1]
#     save_path = f"./asdsafasd_{entry}.jpg"
#     cv2.imwrite(save_path, sum_map)
#     compose_heatmap_image(sum_map, dimg, save_path=f"./sum_{entry}.jpg")
    

#     pass"""