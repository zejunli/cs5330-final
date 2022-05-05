"""
    Team member: Luofei Shi and Zejun Li
    File: cnn_hgr.py
    Description: This file loads, prepares and feeds the data to our network,
                 and it trains the network over the pre-defined epochs, finally 
                 the plot of loss change over the number of training examples 
                 will show and the trained network will be saved to the current path,
                 this file also contains the functionality of producing the confusion
                 matrix using the trained network and the test dataset.
"""


from distutils.command.build import build
from multiprocessing import reduction
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sympy import false
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sn


"""
    global variables
"""
IMG_WIDTH = 128
IMG_HEIGHT = 48

epochs = 10
learning_rate = 0.01
momentum = 0.0
batch_size_train = 64
batch_size_test = 1000
train_loss = []
train_counter = []
test_loss = []
test_counter = []

X, Y = [], []
x_train, x_test, y_train, y_test = [], [], [], []


"""
    This function loads 20000 images from the dataset and
    make random split(0.7, 0.3) to generate training set and test set.
"""
def load_data():
    global X, Y, x_train, x_test, y_train, y_test
    print("Loading image paths...")
    all_image_paths = []
    for i in range(10):
        root = './leapGestRecog/0' + str(i)
        dirs = os.listdir(root)
        for dir in dirs:
            path = root + '/' + dir
            img_paths = os.listdir(path)
            for p in img_paths:
                all_image_paths.append(path + '/' + p)

    print("Loading images...")

    for path in all_image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        X.append(img)

        label = int(path.split('/')[3].split('_')[0][1])
        Y.append(label)

    X = np.array(X, dtype='uint8')
    X = X.reshape(len(all_image_paths), IMG_HEIGHT, IMG_WIDTH, 1)
    Y = np.array(Y)

    print('Images loaded: ', len(X))
    print('Labels loaded: ', len(Y))


    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


"""
    This is a customized Dataset class for feeding dataloader
"""
class HandGestureDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, training=True, transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.data = data
        self.labels = labels

    """
        returns the number of data entries in this dataset
    """
    def __len__(self):
        return len(self.data)

    """
        returns a data entry when indexed, it also applies the passed-along
        transformation
    """
    def __getitem__(self, index):
        return self.transform(self.data[index]), torch.tensor(self.labels[index])



"""
    This is the network itself, this class
    defines the network and the forward calculation
    procedures.
"""
class Network(nn.Module):
    def __init__(self, ):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=.5)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3)
        # 64 * 4 * 14 = 3584
        self.fc1 = nn.Linear(3584, 128)
        self.fc2 = nn.Linear(128, 10)

    """
        computes a forward pass for the network
        methods need a summary comment
    """
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 3584)
        x = F.relu(self.fc1(x))

        return F.log_softmax(self.fc2(x))


"""
    This function trains the network with given dataloader,
    optimizer, and epochs
"""
def train_network(network, train_loder, optimizer, epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loder):
        optimizer.zero_grad()
        # forward
        output = network(data)
        # loss function negative likelihood loss
        loss = F.nll_loss(output, target)
        loss.backward()
        # back propagation
        optimizer.step()

        # print out loss periodically
        if batch_idx % 10 == 0:
            info = 'Train epoch: {} [{}/{} ({:.0f})]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loder.dataset),
                100.0 * batch_idx / len(train_loder), loss.item()
            )
            print(info)
            train_loss.append(loss.item())
            train_counter.append(
                batch_idx * batch_size_train + (epoch - 1) * len(train_loder.dataset)
            )


"""
    This function tests the network with the data in the 
    given dataloader
"""
def test_model(network, test_loader):
    network.eval()
    t_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            t_loss += F.nll_loss(output, target, reduction='sum').item()
            prediction = output.data.max(1, keepdim=True)[1]
            correct += prediction.eq(target.data.view_as(prediction)).sum()

    # average loss
    t_loss /= len(test_loader.dataset)
    test_loss.append(t_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        t_loss, correct, len(test_loader.dataset),
        100.0 * correct / len(test_loader.dataset)
    ))



"""
    This function uses everything defined above to train
    and test the network, once the network training is done,
    the learning model is saved to the current path.
"""
def main():
    # to make the code repeatable
    # torch.manual_seed(100)
    # torch.backends.cudnn.enabled = False

    # initialize network
    network = Network()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.Adam(network.parameters(), lr=learning_rate) # very bad performance
    
    root = './leapGestRecog'
    training_ds = HandGestureDataset(x_train, y_train, training=True, 
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ]))
    test_ds = HandGestureDataset(x_test, y_test, training=False, 
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ]))
    training_loader = torch.utils.data.DataLoader(training_ds, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size_test, shuffle=True)

    print('\n\n\n\nData is ready!\n\n\n\n')
    test_model(network, test_loader)
    for i in range(1, epochs + 1):
        train_network(network, training_loader, optimizer, i)
        test_model(network, test_loader)

    # torch.save(network.state_dict(), './model.pt')

    test_counter = [i * len(training_loader.dataset) for i in range(epochs + 1)]
    fig = plt.figure()
    plt.plot(train_counter, train_loss, color='blue')
    plt.scatter(test_counter, test_loss, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('# training examples')
    plt.ylabel('negative log likelihood loss')
    plt.show()


"""
    This function produces a confusion matrix using the 
    learned model on the test dataset.
"""
def build_confusion():
    network = Network()
    network.load_state_dict(torch.load('./model.pt'))

    network.eval()

    test_ds = HandGestureDataset(x_test, y_test, training=False, 
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ]))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=True)
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            output = network(inputs)
            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)

    classes = ('down', 'palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c')

    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes], columns = [i for i in classes])
    plt.figure(figsize = (12, 7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.savefig('./confusion_matrix.png')


"""
    entry point
"""
if __name__ == '__main__':
    load_data()
    main()
    # build_confusion()