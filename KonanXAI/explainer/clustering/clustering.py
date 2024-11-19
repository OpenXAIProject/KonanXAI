import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable

from typing import Literal, List, Optional, Callable, Union

import os
import json
import numpy as np
import cv2

import torchvision
import torchvision.models as models
from torchvision import transforms
from torchvision import datasets

from KonanXAI.utils.h5file import create_dataset, append_sample, create_attribution_database, append_attributions
from KonanXAI.attribution import GradCAM
import darknet


from PIL import Image
import matplotlib.pyplot as plt
import h5py

from corelay.base import Param
from corelay.processor.base import Processor
from corelay.processor.flow import Sequential, Parallel
from corelay.pipeline.spectral import SpectralClustering
from corelay.processor.clustering import KMeans
from corelay.processor.embedding import TSNEEmbedding, EigenDecomposition


class Flatten(Processor):
    def function(self, data):
        return data.reshape(data.shape[0], np.prod(data.shape[1:]))

class SumChannel(Processor):
    axis = Param(int,1)
    def function(self, data):
        return data.sum(1)

class Normalize(Processor):
    def function(self, data):
        data = data / data.sum((1,2), keepdims = True)
        return data

pipeline = SpectralClustering(
    preprocessing=Sequential([
        SumChannel(),
        Normalize(),
        Flatten()
    ]),
    embedding=EigenDecomposition(n_eigval=5, is_output=True),
    clustering=Parallel([
        Parallel([
            KMeans(n_clusters=number_of_clusters) for number_of_clusters in range(2, 31)
        ], broadcast=True),
        TSNEEmbedding()
    ], broadcast=True, is_output=True)
)

number_of_clusters_list = range(2,31)


# ABCMeta 상속으로 해야하나?
class SpectralClustering:
    ''' explain something...
    
    '''
    def __init__(self, 
                 framework,
                 model,
                 dataset,
                 config):
        self.framework = framework
        self.model = model
        self.dataset = dataset
        self.len_dataset = len(self.dataset)
        self.attribution_algorithm = config['explainer']
        self.h5_dataset_file_path = config['h5_dataset_file_path']
        self.h5_attribution_file_path = config['h5_attr_file_path']
        self.label_json_path = config['label_json_path']
        



    def load_dataset_h5file(self):
        pass
        

    def write_project_script(self):
        pass


    def load_label(self):
        with open(self.label_json_path, 'r', encoding='utf-8') as label_map_file:
            label_map = json.load(label_map_file)
            label_dict = {label['name']:label['index'] for label in label_map}

        


    def apply(self):
        with open(self.label_json_path, 'r', encoding='utf-8') as label_map_file:
            label_map = json.load(label_map_file)
            wordnet_id_map = {label['index']: label['word_net_id'] for label in label_map}

        with h5py.File(self.h5_attribution_file_path) as attribution_file:
            labels = attribution_file['label'][:]

        for class_index in range(5):
            with h5py.File(self.h5_attribution_file_path, 'r') as attribution_file:
                indices_of_samples_in_class, = np.nonzero(labels == class_index)
                attribution_data = attribution_file['attribution'][indices_of_samples_in_class, :]

            (eigenvalues, embedding), (kmeans, tsne) = pipeline(attribution_data)

            with h5py.File(self.h5_attribution_file_path[:-2] + f'_{class_index}.h5', 'w') as analysis_file:
                analysis_name = wordnet_id_map[class_index]

                analysis_group = analysis_file.require_group(analysis_name)
                analysis_group['index'] = indices_of_samples_in_class.astype('uint32')

                embedding_group = analysis_group.require_group('embedding')
                embedding_group['spectral'] = embedding.astype(np.float32)
                embedding_group['spectral'].attrs['eigenvalue'] = eigenvalues.astype(np.float32)

                embedding_group['tsne'] = tsne.astype(np.float32)
                embedding_group['tsne'].attrs['embedding'] = 'spectral'
                embedding_group['tsne'].attrs['index'] = np.array([0, 1])

                cluster_group = analysis_group.require_group('cluster')
                for number_of_clusters, clustering in zip(number_of_clusters_list, kmeans):
                    clustering_dataset_name = f'kmeans-{number_of_clusters:02d}'
                    cluster_group[clustering_dataset_name] = clustering
                    cluster_group[clustering_dataset_name].attrs['embedding'] = 'spectral'
                    cluster_group[clustering_dataset_name].attrs['k'] = number_of_clusters
                    cluster_group[clustering_dataset_name].attrs['index'] = np.arange(
                        embedding.shape[1],
                        dtype=np.uint32
                    )


