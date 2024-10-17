import torchvision.transforms as transforms
__all__ = ["convert_tensor"]
def convert_tensor(images, data_type, img_size):
    if data_type == "imagenet":
        normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std = [0.229, 0.224,0.225])
    else:
        normalize = transforms.Normalize(mean=[0.,0.,0.],std = [1., 1., 1.])
    tensor =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size, antialias=False),
        normalize
    ])
    return tensor(images)