from lime import lime_image
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from ..utils.segment_wrapper import SegmentationAlgorithm
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
        self.config = config
        self.seg_param = config['segments']
        if framework == "darknet":
            self.input = input
            self.input_size = self.input.shape
        else:
            self.input = input[4]
        
    def calculate(self):
        image = self.convert_resize(self.input)
        seg_type = SegmentationAlgorithm(**self.seg_param)
        explainer = lime_image.LimeImageExplainer(random_state=self.config['seed'])
        explanation = explainer.explain_instance(np.array(image),
                                                 self.batch_predict,
                                                 top_labels = 5,
                                                 hide_color = 0,
                                                 num_samples = self.config['num_samples'],
                                                 random_seed = self.config['seed'],
                                                 segmentation_fn = seg_type)
        img, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only = self.config['positive_only'], num_features = self.config['num_features'], hide_rest = self.config['hide_rest'])
        img_boundry = mark_boundaries(img/255., mask)
        return img_boundry

    def convert_tensor(self, images):
        torchvision_models = ['resnet50', 'resnet18', 'vgg16', 'vgg19', 'efficientnet_b0']
        if self.model_name in torchvision_models:
            normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std = [0.229, 0.224,0.225])
        else:
            normalize = transforms.Normalize(mean=[0.,0.,0.],std = [1., 1., 1.])
        tensor =  transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        return tensor(images)
        
    def convert_resize(self, images):
        resize = transforms.Compose([transforms.Resize((224,224))])
        return resize(images)
    
    def batch_predict(self, images):
        self.model.eval()
        batch = torch.stack(tuple(self.convert_tensor(image) for image in images), dim = 0).to(self.device)
        logits = self.model(batch)
        probs = F.softmax(logits, dim = 1)
        return probs.detach().cpu().numpy()
