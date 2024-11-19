import torch
import torchvision.transforms as transforms
import numpy as np

from KonanXAI.utils.data_convert import convert_tensor
from ..utils.segment_wrapper import SegmentationAlgorithm
import shap
from tqdm import tqdm
import matplotlib.pyplot as plt
__all__ = ["KernelShap"]
class KernelShap:
    def __init__(
            self, 
            framework, 
            model, 
            input, 
            config):
        '''
        input: [batch, channel, height, width] torch.Tensor 
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.framework = framework
        self.model = model.to(self.device)
        self.model_name = self.model.model_name
        self.config = config
        self.label_index = None
        self.seg_param = config['segments']
        if framework == "darknet":
            self.input = input
            self.input_size = self.input.shape
        else:
            self.input = input[4]
            
    def calculate(self,inputs=None,targets=None):
        if inputs != None:
            self.input = inputs
        if targets != None:
            self.label_index = targets
        self.img = np.array(self.convert_resize(self.input))
        self.input_size = self.img.shape[0:2]
        self.segment = SegmentationAlgorithm(**self.seg_param)(self.img)
        with torch.no_grad():
            explainer = shap.KernelExplainer(self.infer_mask, np.zeros((1,50)))
            shap_values = explainer.shap_values(np.ones((1,50)), nsamples = self.config['nsamples'], gc_collect = True)
            preds = self.model(convert_tensor(self.img,self.data_type,self.input_size).unsqueeze(0).to(self.device)).detach().cpu().numpy()
        if self.label_index == None:
            top_preds = np.argsort(-preds)
            result = self.fill_segmentation(shap_values[0, :, top_preds[0][0]], self.segment)
        else:
            result = self.fill_segmentation(shap_values[0, :, self.label_index], self.segment)
        return result
        
    def convert_resize(self, images):
        resize = transforms.Compose([transforms.Resize((224,224))])
        return resize(images)
    
    def mask_image(self, zs, segmentation, image, background=None):
        if background is None:
            background = image.mean((0,1))
        out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
        for i in range(zs.shape[0]):
            out[i,:,:,:] = image
            for j in range(zs.shape[1]):
                if zs[i,j] == 0:
                    out[i][segmentation == j,:] = background
        return out
    
    def infer_mask(self, z):
        image = self.mask_image(z, self.segment, self.img, 1)
        if image.max()> 1:
            image = image / image.max()
        image = torch.stack(tuple(convert_tensor(i,self.data_type,self.input_size) for i in image), dim=0)
        image = image.type(torch.float32).to(self.device) #torch.tensor(image, dtype=torch.float32, device="cuda")
        out = self.model(image)
        return out.detach().cpu().numpy()
    
    def fill_segmentation(self, values, segmentation):
        out = np.zeros(segmentation.shape)
        for i in range(len(values)):
            out[segmentation == i] = values[i]
        return out