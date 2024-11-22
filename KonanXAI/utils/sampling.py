import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
from tqdm import tqdm
#from darknet.yolo import BBox
from torchvision.utils import save_image

def gaussian_noise(sample_shape, std, num_samples):
    zeros = torch.zeros(sample_shape)
    noised_sample = torch.repeat_interleave(zeros, num_samples, dim=0)
    for i in range(num_samples):
        noised_sample[i] = zeros.normal_(std=std)
    return noised_sample
    
