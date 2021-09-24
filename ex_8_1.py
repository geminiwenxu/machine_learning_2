import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler
import torch.optim as optim


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Train:
    def __init__(self, net, criterion, opt, device):
        self.net = net
        self.criterion = criterion
        self.opt = opt
        self.device = device

    def count_correct(self, y_pred, y):
        return torch.sum(torch.argmax(y_pred, dim=1) == y)

    def predict(self, X):
        self.net.eval()
        return self.net(X)

    def train(self, trainloader, epochs, testloader=None, verbose=1):
        losses = []
        accs = []
        test_losses = []
        test_accs = []
        for epoch in range(epochs):
            self.net.train()

            correct_predictions = 0
            total_loss = 0

            for i, data in enumerate(trainloader):
                X, y = data[0].to(self.device), data[1].to(self.device)
                y_pred = self.net(X)
                self.opt.zero_grad()
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.opt.step()

                correct_predictions += self.count_correct(y_pred, y).item()
                total_loss += loss.item()

            total_loss /= len(trainloader)
            correct_predictions /= (len(trainloader) * trainloader.batch_size)

            losses.append(total_loss)
            accs.append(correct_predictions)

            if testloader is not None:
                test_loss, test_acc = self.validate(testloader)
                test_losses.append(test_loss)
                test_accs.append(test_acc)

            if verbose == 1 and testloader is not None:
                print("Epoch: {}; Loss: {}; Acc: {}; Test Loss: {}; Test Acc: {}"
                      .format(epoch, total_loss, correct_predictions, test_loss, test_acc))
            elif verbose == 1:
                print("Epoch: {}; Loss: {}; Acc: {}".format(epoch, total_loss, correct_predictions))

        return losses, accs, test_losses, test_accs

    def validate(self, testloader):
        self.net.eval()
        total_loss = 0
        correct_predictions = 0
        for i, data in enumerate(testloader):
            X, y = data[0].to(self.device), data[1].to(self.device)
            y_pred = self.net(X)
            loss = self.criterion(y_pred, y)

            total_loss += loss.item()
            correct_predictions += self.count_correct(y_pred, y).item()

        total_loss /= len(testloader)
        correct_predictions /= (len(testloader) * testloader.batch_size)

        return total_loss, correct_predictions


if __name__ == '__main__':
    # configurations
    lr = 0.01
    batch_size = 128
    epochs = 50
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # preparing train and test dataloaders
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_indices = list(range(int(len(trainset)/100)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2,
                                              sampler=SubsetRandomSampler(train_indices))
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

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

    # plotting test/train accuracy
    plt.title('accuracy')
    plt.plot(train_accs), plt.plot(test_accs)
    plt.legend(['train', 'test'])
    plt.show()
