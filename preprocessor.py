
import os
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np


# customized dataset
class HandGestureDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, training=True, transform=None) -> None:
        super().__init__()
        if training:
            self.data, self.labels = self.get_training_data(root_path)
        else:
            self.data, self.labels = self.get_test_data(root_path)
        
        self.transform = transform


    # load training data
    def get_training_data(self, root_path):
        data, labels = [], []
        # 00 - 08 for training, 09 for test
        for i in range(9):
            path = root_path + '/0' + str(i)
            dir_with_type = os.listdir(path)
            for dir in dir_with_type:
                label = int(dir.split('_')[0])
                full_type_path = path + '/' + dir
                images = os.listdir(full_type_path)
                for name in images:
                    full_image_path = full_type_path + '/' + name
                    data.append(Image.open(full_image_path))
                    labels.append(label)

        return data, labels


    # load test data
    def get_test_data(self, root_path):
        data, labels = [], []
        path = root_path + '/09'
        dir_with_type = os.listdir(path)
        for dir in dir_with_type:
            label = int(dir.split('_')[0])
            full_type_path = path + '/' + dir
            images = os.listdir(full_type_path)
            for name in images:
                full_image_path = full_type_path + '/' + name
                data.append(Image.open(full_image_path))
                labels.append(label)
        return data, labels
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]





# preprocess all images
def main():
    training_ds = HandGestureDataset('./leapGestRecog', training=True)
    print(len(training_ds))

    # ds = torchvision.datasets.MNIST('./data/', train=True, download=True)
    
    # data, label = ds[0]
    # print(data)
    # print(label)

    # img = Image.open('./leapGestRecog/00/01_palm/frame_00_01_0001.png')
    # print(img)


if __name__ == '__main__':
    main()