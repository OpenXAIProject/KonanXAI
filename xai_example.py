import KonanXAI as XAI
from KonanXAI.lib.core import darknet
from KonanXAI.lib.attribution.lrp.blocks import Graph
import cv2
import numpy as np
import yaml
# model 
mtype = XAI.ModelType.ResNet50
# platform
platform = XAI.PlatformType.Pytorch
# dataset
dtype = XAI.DatasetType.CUSTOM
# explain
etype = XAI.ExplainType.GradCAM

# with open('config.yaml') as f:
    

xai = XAI.XAI()
# xai.load_model_support(mtype, platform, weight_path="./ckpt/darknet/yolov4_4582/4582")
xai.load_model_support(mtype, platform, weight_path="./resnet50-0676ba61.pth")
# xai.load_dataset_support(dtype, maxlen=10,path = "D:/Datasets/cifar10", fit_size=(640,640))
xai.load_dataset_support(dtype, maxlen=10,path = "./data", fit_size=(224,224))

xai.set_explain_mode([XAI.ExplainType.GradCAM])
explain = xai.explain(target_layer= xai.model.net.layer4[-1].relu)
# explain = xai.explain()
explain.save_heatmap("heatmap/")

# a = Graph(xai.model, "C:/Users/KsKim/.cache/torch/hub/ultralytics_yolov5_master/models/yolov5s.yaml")


# img = darknet.open_image(r"test.jpg", (416, 416))
# net: darknet.Network = xai.model.net
# results = net.predict_using_gradient_hook(img)