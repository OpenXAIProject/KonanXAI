import KonanXAI as XAI
from KonanXAI.lib.core import darknet
from KonanXAI.lib.attribution.lrp.blocks import Graph
import cv2
import numpy as np

# model 
mtype = XAI.ModelType.Yolov5s
# platform
platform = XAI.PlatformType.Pytorch
# dataset
dtype = XAI.DatasetType.COCO
# explain
etype = XAI.ExplainType.GradCAM

xai = XAI.XAI()
xai.load_model_support(mtype, platform, pretrained=True)
# xai.load_dataset_support(dtype, maxlen=10, shuffle=False)

#print(xai.model.net)

a = Graph(xai.model, "C:/Users/KsKim/.cache/torch/hub/ultralytics_yolov5_master/models/yolov5s.yaml")


# img = darknet.open_image(r"test.jpg", (416, 416))
# net: darknet.Network = xai.model.net
# results = net.predict_using_gradient_hook(img)