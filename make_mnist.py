import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt


train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)


training_image_compressed_file = './data/MNIST/raw/train-images-idx3-ubyte.gz'
training_label_compressed_file = './data/MNIST/raw/train-labels-idx1-ubyte.gz'

training_image_decompressed_file = './data/MNIST/raw/train-images-idx3-ubyte'
training_label_decompressed_file = './data/MNIST/raw/train-labels-idx1-ubyte'

training_folder = './data/MNIST/trainset'
training_label_file = './data/MNIST/trainset/labels.txt'

if not os.path.exists(training_folder):
    os.makedirs(training_folder)


with open(training_label_decompressed_file, 'rb') as f:
    f.read(8)
    buf = f.read()

training_labels = np.frombuffer(buf, dtype=np.uint8)

with open(training_label_file, 'w') as f:
    for i in range(60000):
        image_path = f'{training_folder}/{training_labels[i]}/image_{i}.png'
        label_value = training_labels[i]

        f.write(f'{image_path}\t{label_value}\n')

with open(training_image_decompressed_file, 'rb') as f:
    f.read(16)
    buf = f.read()

training_images = np.frombuffer(buf, dtype=np.uint8).reshape(60000, 28,28)

for i, image in enumerate(training_images):
    label_value = training_labels[i]

    digit_folder = os.path.join(training_folder, str(label_value))
    if not os.path.exists(digit_folder):
        os.makedirs(digit_folder)

    image_path = os.path.join(digit_folder, f'image_{i}.png')
    plt.imsave(image_path, image, cmap='gray')