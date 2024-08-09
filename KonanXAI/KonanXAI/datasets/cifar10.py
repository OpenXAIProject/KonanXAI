from KonanXAI.datasets import Datasets
import darknet
import os
from glob import glob
import pickle
import numpy as np
from tqdm import tqdm
import cv2


class CIFAR10(Datasets):
    def __init__(self, src_path):
        super().__init__(src_path)
        self.src_path = src_path
        self.classes = 10
        self.make_cache = {}
        
    def load_src_path(self):
        train_path = glob(self.src_path+"/cifar-10-batches-py")
        test_path = glob(self.src_path+"/cifar-10-batches-py")# 수정 필요
        self.train_items = []
        self.test_items = []
        
        for path in train_path:
            with open(path + "/batches.meta", "rb") as f:
                binary = pickle.load(f, encoding='latin1')
            # 레이블 이름
            word_labels = binary['label_names']
            loaded_image = []
            loaded_image_name = []
            loaded_label = []
            data_batches = glob(path + "/data_batch*")
            for batch in data_batches:
                with open(batch, "rb") as f:
                    binary = pickle.load(f, encoding='bytes')
                loaded_image.append(binary[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
                loaded_image_name.append(binary[b'filenames'])
                loaded_label.append(binary[b'labels'])
            loaded_image = np.concatenate(loaded_image, axis=0)
            loaded_image_name = np.concatenate(loaded_image_name, axis= 0)
            loaded_label = np.concatenate(loaded_label, axis=0)

            for (image_name, numpy_image, label_idx) in tqdm(zip(loaded_image_name, loaded_image, loaded_label)):
                label = label_idx#word_labels[label_idx]
                self.train_items.append(((image_name,numpy_image), label))

        for path in test_path:
            with open(path + "/batches.meta", "rb") as f:
                binary = pickle.load(f, encoding='latin1')
            # 레이블 이름
            word_labels = binary['label_names']
            loaded_image = []
            loaded_label = []
            data_batches = glob(path + "/test_batch")
            for batch in data_batches:
                with open(batch, "rb") as f:
                    binary = pickle.load(f, encoding='bytes')
                loaded_image.append(binary[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
                loaded_label.append(binary[b'labels'])
            loaded_image = np.concatenate(loaded_image, axis=0)
            loaded_label = np.concatenate(loaded_label, axis=0)
            for (numpy_image, label_idx) in tqdm(zip(loaded_image, loaded_label)):
                label = label_idx#word_labels[label_idx]
                self.test_items.append((numpy_image, label))
