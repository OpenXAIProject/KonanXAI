from KonanXAI.attribution.gradcam import GradCAM
from torch.nn import ReLU, SiLU
import torch
import torch.nn.functional as F
def relu_hook_function(m, in_grad, out_grad):
    if isinstance(m,ReLU) or isinstance(m,SiLU):
        return (torch.clamp(in_grad[0],min=0.0),)
def first_layer_hook_function(m, in_grad, out_grad):
        m.first_layer_in_gradients.insert(0, in_grad[0])
        m.first_layer_out_gradients.insert(0,out_grad[0])
class GuidedGradCAM(GradCAM):
    def first_hook_layers(self):
        self.first_layer = list(self.model.children())[0].to(self.device)
        if "yolo" in self.model_name:
            self.first_layer = self.first_layer[0].conv
        elif "resnet" in self.model_name:
            self.first_layer = self.first_layer
        else:
            self.first_layer = self.first_layer[0]
        self.first_layer.first_layer_in_gradients = []
        self.first_layer.first_layer_out_gradients = []
        g_fwd_handle = self.first_layer.register_backward_hook(first_layer_hook_function)
        return g_fwd_handle
    
    def update_relus(self):
        model_handle = []
        for module in self.model.modules():
            if isinstance(module,ReLU) or isinstance(module, SiLU):
                model_handle.append(module.register_backward_hook(relu_hook_function))
        return model_handle
    
    def generate_gradients(self, score):
        gradients_as_arrs = []
        self.model.zero_grad()
        score.backward(retain_graph=True)
        guided_grad = self.first_layer.first_layer_in_gradients[0]
        gradients_as_arr = guided_grad.detach().cpu().data.numpy()
        gradients_as_arr = gradients_as_arr[0, :, :, :]
        gradients_as_arrs.append(gradients_as_arr.transpose((1, 2, 0)))
        return gradients_as_arrs
    
    def _yolo_backward_pytorch(self):
        self.bboxes = []
        self.label_index = []
        self.guided_image = []
        for cls, sel_layer, sel_layer_index in zip(self.pred[0], self.select_layers, self.index_tmep):
            self.model.zero_grad()
            score = self.logits_origin[sel_layer][int(cls[5].item())]
            self.guided_image.append(self.generate_gradients(score))
            for handle in self.model_handle:
                handle.remove()    
            self.model.zero_grad()
            score.backward(retain_graph=True)
            layer = self.layer[sel_layer_index]
            feature = layer.fwd_out[-1].unsqueeze(0)
            gradient = layer.bwd_out[0]
            self.feature.append(feature)
            self.gradient.append(gradient)
            self.label_index.append(int(cls[5].item()))
            self.bboxes.append(cls[...,:4].detach().cpu().numpy())
            
    def get_feature_and_gradient(self):
        guided_image = None
        first_hook_handel = self.first_hook_layers()
        self.feature = []
        self.gradient = []
        self.model_handle = self.update_relus()
        if self.framework == 'torch':
            self.model.eval()
            self.input.requires_grad = True
            self._get_target_layer()
            fwd_handle, bwd_handle = self.set_model_hook()
            
            if self.model_name in ('yolov4', 'yolov4-tiny', 'yolov5s'):
                self._yolo_get_bbox_pytorch()
                self._yolo_backward_pytorch()

            else:
                if self.model.model_algorithm == 'abn':
                    self.att, self.pred, _ = self.model(self.input)
                else:
                    self.pred = self.model(self.input)
                self.label_index = torch.argmax(self.pred).item()
                score = self.pred[0][self.label_index]
                self.guided_image = self.generate_gradients(score)
                for handle in self.model_handle:
                    handle.remove()    
                self.model.zero_grad()
                score.backward(retain_graph = True)
                feature = self.layer.fwd_in[-1]
                gradient = self.layer.bwd_in[-1]
                self.feature.append(feature)
                self.gradient.append(gradient)
                fwd_handle.remove()
                bwd_handle.remove()
                return self.feature, self.gradient

        elif self.framework == 'darknet':
            print("Not supported Framewokr")
            # self.model.forward_image(self.input)
            # self._yolo_get_bbox_darknet()
            # self._yolo_backward_darknet()
            # return self.feature, self.gradient
    
    
    def calculate(self):
        self.get_feature_and_gradient()
        self.heatmaps = [] 
        for feature, gradient in zip(self.feature, self.gradient):
            b, ch, h, w = gradient.shape
            alpha = gradient.reshape(b, ch, -1).mean(2)
            weights = alpha.reshape(b, ch, 1, 1)
            heatmap = (weights * feature).sum(1, keepdim=True)
            heatmap = F.relu(heatmap)
            self.heatmaps.append(heatmap)
            
        if self.model_name[0:4] == 'yolo':
            return self.heatmaps, self.bboxes, self.guided_image
        else:
            return self.heatmaps, [self.guided_image]