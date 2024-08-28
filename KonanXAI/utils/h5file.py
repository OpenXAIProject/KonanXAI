import darknet  
# from ..lib.core import dtrain
# from ..lib.core import pytorch
import random
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import h5py


def create_dataset(dataset_file_path, samples_shape, number_of_samples):

    dataset_file = h5py.File(dataset_file_path, 'w')
    dataset_file.create_dataset(
        'data',
        shape = (number_of_samples,) + tuple(samples_shape),
        dtype = 'float32'
    )
    dataset_file.create_dataset(
        'label',
        shape = (number_of_samples,),
        dtype = 'uint16'
    )
    return dataset_file

def append_sample(dataset_file, index, sample, label):
    dataset_file['data'][index:sample.shape[0]+index] = sample
    dataset_file['label'][index:sample.shape[0]+index] = label


def create_attribution_database(
    attribution_database_file_path,
    attribution_shape,
    number_of_classes,
    number_of_samples):
    
    attribution_database_file = h5py.File(attribution_database_file_path, 'w')
    attribution_database_file.create_dataset(
        'attribution',
        shape = (number_of_samples,) + tuple(attribution_shape),
        dtype = 'float32'
    )
    attribution_database_file.create_dataset(
        'prediction',
        shape = (number_of_samples, number_of_classes),
        dtype = 'float32'
    )
    attribution_database_file.create_dataset(
        'label',
        shape = (number_of_samples,),
        dtype = 'uint16'
    )
    return attribution_database_file

def append_attributions(
    attribution_database_file,
    index,
    attributions,
    predictions,
    labels):

    attribution_database_file['attribution'][index:attributions.shape[0]+index] = (
      attributions.detach().numpy())
    attribution_database_file['prediction'][index:predictions.shape[0] + index] = (
      predictions.detach().numpy())
    attribution_database_file['label'][index:labels.shape[0] + index] = (
      labels.detach().numpy())
