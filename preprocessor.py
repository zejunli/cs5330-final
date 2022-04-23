
import os
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image


# preprocess all images
def main():
    print(os.listdir('./leapGestRecog'))

    data = []
    for i in range(10):
        path = './leapGestRecog/0' + str(i)
        dir_with_type = os.listdir(path)
        for dir in dir_with_type:
            label = int(dir.split('_')[0])
            full_type_path = path + '/' + dir
            images = os.listdir(full_type_path)
            for name in images:
                full_image_path = full_type_path + '/' + name
                data.append(Image.open(full_image_path))
                print(full_image_path)




    # ds = torchvision.datasets.MNIST('./data/', train=True, download=True)
    
    # data, label = ds[0]
    # print(data)
    # print(label)

    # img = Image.open('./leapGestRecog/00/01_palm/frame_00_01_0001.png')
    # print(img)


if __name__ == '__main__':
    main()