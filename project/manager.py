import yaml
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import KonanXAI as XAI
class Configuration:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(os.path.dirname(__file__)+self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self._parser_config()
        self.xai = XAI.XAI()
    
    def _parser_config(self):
        self.save_path = self.config['config']['save_path']
        self.weight_path = self.config['config']['weight_path']
        self.data_path = self.config['config']['data_path']
        self.data_resize = self.config['config']['data_resize']
        self.model_type = self.config['config']['model_type']
        self.platform_type = self.config['config']['platform_type']
        self.data_type = self.config['config']['data_type']
        self.explain_type = self.config['config']['explain_type']

    def get_target_layer(self, model):
        if self.platform_type.lower() == "pytorch":
            self.target_layer = model
            target = self.config['config']['target_layer']
            if isinstance(target,list):
                for layer in target:
                    self.target_layer = self.target_layer._modules[layer]
            elif isinstance(target,dict):
                self.target_yolo_layer = []
                for index, layers in target.items():
                    base_layer = model
                    for layer in layers:
                        base_layer = base_layer._modules[layer]
                    self.target_yolo_layer.append(base_layer)
                self.target_layer = self.target_yolo_layer
            else:
                self.target_layer = None
                
        elif self.platform_type.lower() == "darknet":
            self.target_layer = None
        
    def _check_model_name(self):
        if self.model_type.lower() == "resnet50":
            return XAI.ModelType.ResNet50
        elif self.model_type.lower() == "vgg16":
            return XAI.ModelType.VGG16
        elif self.model_type.lower() == "vgg19":
            return XAI.ModelType.VGG19
        elif self.model_type.lower() == "efficientnet_b0":
            return XAI.ModelType.EfficientNet_B0
        elif self.model_type.lower() == "yolov4":
            return XAI.ModelType.Yolov4
        elif self.model_type.lower() == "yolov4tiny":
            return XAI.ModelType.Yolov4Tiny
        elif self.model_type.lower() == "yolov5s":
            return XAI.ModelType.Yolov5s
        elif self.model_type.lower() == "custom":
            return XAI.ModelType.Custom
        else:
            raise Exception('Model_type_None')
        
    def _check_platform_type(self):
        if self.platform_type.lower() == "pytorch":
            return XAI.PlatformType.Pytorch
        elif self.platform_type.lower() == "darknet":
            return XAI.PlatformType.Darknet
        elif self.platform_type.lower() == "dtrain":
            return XAI.PlatformType.DTrain
        else:
            raise Exception("Platform_type_None")
        
    def _check_data_type(self):
        if self.data_type.lower() == "mnist":
            return XAI.DatasetType.MNIST
        elif self.data_type.lower() == "coco":
            return XAI.DatasetType.COCO
        elif self.data_type.lower() == "custom":
            return XAI.DatasetType.CUSTOM
        elif self.data_type.lower() == "cifar10":
            return XAI.DatasetType.CIFAR10
        else:
            raise Exception("Dataset_type_None")
        
    def _check_explain_type(self):
        if self.explain_type.lower() == "gradcam":
            return XAI.ExplainType.GradCAM
        elif self.explain_type.lower() == "gradcampp":
            return XAI.ExplainType.GradCAMpp
        elif self.explain_type.lower() == "eigencam":
            return XAI.ExplainType.EigenCAM
        elif self.explain_type.lower() == "lrp":
            mode = self._check_explain_mode()
            return XAI.ExplainType.LRP, mode
        elif self.explain_type.lower() == "ig":
            return XAI.ExplainType.IG
        else:
            raise Exception("explain_type_None")
    def _check_explain_mode(self):
        try:
            mode = self.config['config']['explain_mode']
            if mode.lower() == "epslion":
                return XAI.LRPRule.Epsilon
            elif mode.lower() == "alphbeta":
                return XAI.LRPRule.AlphaBeta
        except:
            raise Exception("explain_mode is None")