
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

epochs = 30
learning_rate = 0.001
momentum = 0.9
batch_size_train = 100
batch_size_test = 500
train_loss = []
train_counter = []
test_loss = []
test_counter = []

# customized dataset
class HandGestureDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, training=True, transform=None) -> None:
        super().__init__()
        self.transform = transform
        if training:
            self.data, self.labels = self.get_training_data(root_path)
        else:
            self.data, self.labels = self.get_test_data(root_path)
    

    # load training data
    def get_training_data(self, root_path):
        data, labels = [], []
        count = 1
        # 00 - 08 for training, 09 for test
        for i in range(9):
            path = root_path + '/0' + str(i)
            dir_with_type = os.listdir(path)
            for dir in dir_with_type:
                print("Loading training data, processing folder %s...(%d / 90)" % (dir, count))
                count += 1
                label = int(dir.split('_')[0])
                full_type_path = path + '/' + dir
                images = os.listdir(full_type_path)
                for name in images:
                    full_image_path = full_type_path + '/' + name
                    if self.transform:
                        data.append(np.asarray(Image.open(full_image_path).resize((128, 48))))
                    else:
                        data.append(Image.open(full_image_path))
                    # to prevent index error
                    labels.append(label - 1)

        return data, labels


    # load test data
    def get_test_data(self, root_path):
        data, labels = [], []
        path = root_path + '/09'
        dir_with_type = os.listdir(path)
        count = 1
        for dir in dir_with_type:
            print("Loading test data, processing folder %s...(%d / 10)" % (dir, count))
            count += 1
            label = int(dir.split('_')[0])
            full_type_path = path + '/' + dir
            images = os.listdir(full_type_path)
            for name in images:
                full_image_path = full_type_path + '/' + name
                if self.transform:
                    data.append(np.asarray(Image.open(full_image_path).resize((128, 48))))
                else:
                    data.append(Image.open(full_image_path))
                labels.append(label - 1)
        return data, labels
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.transform(self.data[index]), torch.tensor(self.labels[index])



class Network(nn.Module):
    def __init__(self, ):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=.3)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        # 20 * 29 * 9 = 5220
        self.fc1 = nn.Linear(5220, 50)
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        # convolution1 -> max pool -> relu
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # convolution2 -> drop out -> max pool -> relu
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # flatten -> linear layer with 50 nodes -> relu
        x = x.view(-1, 5220)
        x = F.relu(self.fc1(x))

        # linear layer with 10 nodes -> log_softmax
        x = F.log_softmax(self.fc2(x))

        return x


# useful functions with a comment for each function
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


# preprocess all images
def main():
    # to make the code repeatable
    torch.manual_seed(100)
    torch.backends.cudnn.enabled = False

    # initialize network
    network = Network()
    
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    
    root = './leapGestRecog'
    training_ds = HandGestureDataset(root, training=True, 
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ]))
    test_ds = HandGestureDataset(root, training=False, 
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ]))
    training_loader = torch.utils.data.DataLoader(training_ds, batch_size=batch_size_train, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size_test, shuffle=False)

    print('\n\n\n\nData is ready!\n\n\n\n')
    test_model(network, test_loader)
    for i in range(1, epochs + 1):
        # if i > 15:
        #     for g in optimizer.param_groups:
        #         g['lr'] = learning_rate / (epochs - 5)
        train_network(network, training_loader, optimizer, i)
        test_model(network, test_loader)


    # ds = torchvision.datasets.MNIST('./data/', train=True, download=False, 
    # transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    # train_loader = torch.utils.data.DataLoader(
    #         torchvision.datasets.MNIST('./data/', train=True, download=True, 
    #         transform=torchvision.transforms.Compose([
    #             torchvision.transforms.ToTensor()
    #         ])),
    #         batch_size=1,
    #         shuffle=True
    #     )
    # data, label = ds[0]
    # print(data.shape)

    # a, b = test_ds[0]
    # print(a.shape)


    # img = Image.open('./leapGestRecog/00/01_palm/frame_00_01_0001.png').resize((320, 120))
    # img.show()


if __name__ == '__main__':
    main()