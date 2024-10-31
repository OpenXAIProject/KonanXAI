from lime import lime_image
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

from KonanXAI.utils.data_convert import convert_tensor
from ..utils.segment_wrapper import SegmentationAlgorithm
__all__ = ["LimeImage"]
class LimeImage:
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
        self.label_index = None
        self.config = config
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
            if isinstance(targets,(tuple,list)):
                self.label_index = targets.item()
            else:
                self.label_index = targets
        image = self.convert_resize(self.input)
        self.input_size = image.size
        seg_type = SegmentationAlgorithm(**self.seg_param)
        explainer = lime_image.LimeImageExplainer(random_state=self.config['seed'])
        explanation = explainer.explain_instance(np.array(image),
                                                 self.batch_predict,
                                                 top_labels = -1,
                                                 hide_color = 0,
                                                 num_samples = self.config['num_samples'],
                                                 random_seed = self.config['seed'],
                                                 segmentation_fn = seg_type)
        if self.label_index == None:
            img, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only = self.config['positive_only'], num_features = self.config['num_features'], hide_rest = self.config['hide_rest'])
        else:
            img, mask = explanation.get_image_and_mask(self.label_index, positive_only = self.config['positive_only'], num_features = self.config['num_features'], hide_rest = self.config['hide_rest'])
        img_boundry = mark_boundaries(img/255., mask)
        return img_boundry
        
    def convert_resize(self, images):
        resize = transforms.Compose([transforms.Resize((224,224))])
        return resize(images)
    
    def batch_predict(self, images):
        self.model.eval()
        batch = torch.stack(tuple(convert_tensor(image,self.data_type,self.input_size) for image in images), dim = 0).to(self.device)
        logits = self.model(batch)
        probs = F.softmax(logits, dim = 1)
        return probs.detach().cpu().numpy()
