import KonanXAI as XAI
from KonanXAI.lib.core import darknet
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

print(xai.model.net)
# img = darknet.open_image(r"test.jpg", (416, 416))
# net: darknet.Network = xai.model.net
# results = net.predict_using_gradient_hook(img)