# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class SimpleFCNet(nn.Module):
    """
    A simple neural network with fully connected layers
    """

    def __init__(self, input_shape=(28, 28), num_classes=10):
        super(SimpleFCNet, self).__init__()
        # create the model by adding the layers
        layers = []

        ###################################
        #     fill in the code here       #
        ###################################
        # Add a Flatten layer to convert the 2D pixel array to a 1D vector
        layers.append(nn.Flatten())
        # Add a fully connected / linear layer with 128 nodes
        layers.append(nn.Linear(28 * 28, 128))
        # Add ReLU activation
        layers.append(nn.ReLU(inplace=True))
        # Append a fully connected / linear layer with 64 nodes
        layers.append(nn.Linear(128, 64))
        # Add ReLU activation
        layers.append(nn.ReLU(inplace=True))
        # Append a fully connected / linear layer with num_classes (10) nodes
        layers.append(nn.Linear(64, 10))
        self.layers = nn.Sequential(*layers)

        self.reset_params()

    def forward(self, x):
        # the forward propagation
        out = self.layers(x)
        if self.training:
            # softmax is merged into the loss during training
            return out
        else:
            # attach softmax during inference
            out = nn.functional.softmax(out, dim=1)
            return out

    def reset_params(self):
        # to make our model a faithful replica of the Keras version
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class SimpleConvNet(nn.Module):
    """
    A simple convolutional neural network
    """

    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(SimpleConvNet, self).__init__()

        self.conv1 = ConvBlock(3, 16, kernel_size=3, stride=1, padding=2)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.incep0 = (InceptionBlock(16,8,16,32,4,8,8))
        # self.max4 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = ConvBlock(16, 32, kernel_size=3, stride=1, padding=1)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.incep1 = (InceptionBlock(32, 16, 28, 64, 8, 32, 32))
        self.incep2 = (InceptionBlock(144, 32, 16, 32, 4, 8, 8))
        # self.incep3 = (InceptionBlock(80,32,16,32,8,8,8))
        self.max3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.4)

        # self.incep3 = (InceptionBlock(80,32,16,32,8,8,8))

        # self.max4 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.linear1 = (nn.Linear(1280, 100))

        # self.layers = nn.Sequential(*layers)

    def forward(self, x):
        #################################
        # Update the code here as needed
        #################################
        x = self.conv1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.incep1(x)
        x = self.incep2(x)
        x = self.max3(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        return x


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_1x1pool, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ######################################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ######################################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch + 1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch + 1, 100. * test_acc))

    return test_acc
