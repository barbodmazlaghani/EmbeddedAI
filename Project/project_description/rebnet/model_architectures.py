import torch
from torch.nn import Module
import torch.nn as nn
from binarization_utils import *
import torch.nn.functional as F

batch_norm_eps = 1e-4
batch_norm_alpha = 0.1  #(this is same as momentum)

class CIFAR10Model(Module):
    def __init__(self, input_size=(32,32)):
        super(CIFAR10Model, self).__init__()
        size = input_size[0]

        self.conv1 = nn.Conv2d(3, 64, 3, padding=0)
        self.bn1 = nn.BatchNorm2d(64,
                        momentum=batch_norm_alpha,
                        eps=batch_norm_eps)
        
        size = size - 3 + 1

        self.conv2 = nn.Conv2d(64, 64, 3, padding=0)
        self.bn2 = nn.BatchNorm2d(64,
                        momentum=batch_norm_alpha,
                        eps=batch_norm_eps)

        size = size - 3 + 1

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        size = int(((size - 2) / 2) + 1)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=0)
        self.bn3 = nn.BatchNorm2d(128,
                        momentum=batch_norm_alpha,
                        eps=batch_norm_eps)

        size = size - 3 + 1

        self.conv4 = nn.Conv2d(128, 128, 3, padding=0)
        self.bn4 = nn.BatchNorm2d(128,
                        momentum=batch_norm_alpha,
                        eps=batch_norm_eps)

        size = size - 3 + 1

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        size = int(((size - 2) / 2) + 1)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=0)
        self.bn5 = nn.BatchNorm2d(256,
                        momentum=batch_norm_alpha,
                        eps=batch_norm_eps)

        size = size - 3 + 1

        self.conv6 = nn.Conv2d(256, 256, 3,padding=0)
        self.bn6 = nn.BatchNorm2d(256,
                        momentum=batch_norm_alpha,
                        eps=batch_norm_eps)

        size = size - 3 + 1
        self.size = size

        self.dense1 = nn.Linear(size * size * 256, 512)
        self.bn7 = nn.BatchNorm1d(512,
                        momentum=batch_norm_alpha,
                        eps=batch_norm_eps)

        self.dense2 = nn.Linear(512, 512)
        self.bn8 = nn.BatchNorm1d(512,
                        momentum=batch_norm_alpha,
                        eps=batch_norm_eps)

        self.dense3 = nn.Linear(512, 10)
        self.bn9 = nn.BatchNorm1d(10,
                        momentum=batch_norm_alpha,
                        eps=batch_norm_eps)

    def forward(self, x):
        o = self.conv1(x)
        o = F.relu(self.bn1(o))

        o = self.conv2(o)
        o = F.relu(self.bn2(o))

        o = self.pool1(o)
        
        o = self.conv3(o)
        o = F.relu(self.bn3(o))

        o = self.conv4(o)
        o = F.relu(self.bn4(o))

        o = self.pool2(o)

        o = self.conv5(o)
        o = F.relu(self.bn5(o))

        o = self.conv6(o)
        o = F.relu(self.bn6(o))

        o = o.view(-1, self.size*self.size*256)

        o = self.dense1(o)
        o = F.relu(self.bn7(o))

        o = self.dense2(o)
        o = F.relu(self.bn8(o))

        o = self.dense3(o)
        o = F.relu(self.bn9(o))
        
        return o