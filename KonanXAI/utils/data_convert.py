import torchvision.transforms as transforms
from PIL import Image
__all__ = ["convert_tensor"]
def convert_tensor(images, data_type, img_size):
    if data_type == "imagenet":
        normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std = [0.229, 0.224,0.225])
        tensor =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size, antialias=False),
        normalize
    ])
        #images = tensor(images)
    else:
        tensor =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size, antialias=False),
    ])
        # normalize = transforms.Normalize(mean=[0.,0.,0.],std = [1., 1., 1.])
    if isinstance(images, Image.Image):
        if (data_type == 'mnist') or (data_type == 'counterfactual'):
            images = images.convert('L')
        else:            
            images = images.convert('RGB')
    return tensor(images)