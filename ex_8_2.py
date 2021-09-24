from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
from ex_8_1 import Train, Net
import torchvision
import torch.optim as optim
from PIL import Image
import numpy as np
from skimage import io


class GetDataset(Dataset):
    def __init__(self, data, rotation_degree, transform=None):
        self.data = data
        self.rotation_degree = rotation_degree
        self.transform = transform

    def __len__(self):
        return 4*len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.data[idx]
        image = Image.fromarray(img)
        sample = self.transform(image)
        return (sample, label)


if __name__ == '__main__':
    rotation_degree = [0, 90, 180, 270]
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    transformed_train_dataset = GetDataset(trainset, rotation_degree, transform)
    trainloader = DataLoader(transformed_train_dataset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    transformed_test_dataset = GetDataset(testset, rotation_degree, transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    """ training NN """
    lr = 0.01
    batch_size = 128
    epochs = 50
    num_classes = 4
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # creating instances
    net = Net()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    problem = Train(net, criterion, optimizer, device)

    # let model train
    train_losses, train_accs, test_losses, test_accs = problem.train(trainloader, epochs, testloader, verbose=1)

    # plotting test/train loss
    plt.title('loss')
    plt.plot(train_losses), plt.plot(test_losses)
    plt.legend(['train', 'test'])
    plt.show()
