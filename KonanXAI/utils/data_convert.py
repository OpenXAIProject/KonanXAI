import torchvision.transforms as transforms
from PIL import Image
__all__ = ["convert_tensor"]
def convert_tensor(images, data_type, img_size):
    if data_type == "simagenet":
        normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std = [0.229, 0.224,0.225])
        tensor =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size, antialias=False),
        normalize
    ])
    else:
        tensor =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size, antialias=False),
    ])
        # normalize = transforms.Normalize(mean=[0.,0.,0.],std = [1., 1., 1.])
    images = images.convert('RGB')
    return tensor(images)