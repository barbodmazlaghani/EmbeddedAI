import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
import os
import sys

from tqdm import tqdm

sys.path.insert(0, '.')
from binarization_utils import *
from model_architectures import CIFAR10Model

dataset='CIFAR-10'
Train = True
Evaluate = True
batch_size = 128
epochs = 20

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def preprocess(image):
    img = np.array(image)
    img = img / 255.0
    img = (2 * img - 1).astype(np.float32)
    return img

trs = [
    preprocess,
    transforms.ToTensor()
]
trs.append(transforms.RandomAffine(degrees=0,
                        translate=(0.15, 0.15),
                        scale=None, shear=None, interpolation=False))

trs.append(transforms.RandomHorizontalFlip())


transform = transforms.Compose(trs)

transform2 = transforms.Compose([
    preprocess,
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform2, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=4)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512,
                                           shuffle=False, num_workers=4)

if Train:
    if not (os.path.exists('models')):
        os.mkdir('models')
    if not (os.path.exists('models/' + dataset)):
        os.mkdir('models/' + dataset)

    torch.cuda.set_device(0)

    for resid_levels in range(1, 4):
        # print 'training with', resid_levels,'levels'
        model = CIFAR10Model((32, 32)).to(device)
        prepare(model, True, resid_levels, resid_levels) #convert the model mlp layers to binary
        print(model)
        #gather all binary dense and binary convolution and residual binary activation layers:
        binary_layers = []
        binary_res = []
        for name, l in model.named_modules():
            if isinstance(l, BinaryLinear) or isinstance(l, BinaryConv2d):
                binary_layers.append(l)
            elif isinstance(l, ResidualSign):
                binary_res.append(l)

        lr = 0.01
        criterion = nn.CrossEntropyLoss() #has softmax inside
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

        weights_path = 'models/' + dataset + '/' + str(
            resid_levels) + '_residuals.pt'
        
        for epoch in range(epochs):
            print(f'{epoch}/{epochs}:')
            model.train()
            correct = 0
            total = 0
            running_loss = 0.0
            l = len(train_loader)

            for i, data in tqdm(enumerate(train_loader, 0), leave=True, total=l):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                
                loss = criterion(outputs, labels)

                loss.backward()
                
                optimizer.step()

                optimizer.zero_grad()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_accuracy = 100 * correct / total
            print(f'{epoch + 1}, loss: {(running_loss / len(train_loader)):.3f}, acc: {train_accuracy:.3f}%')
            
            model.eval()
            correct = 0
            running_loss = 0.0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss2 = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    running_loss += loss2.item()

            test_accuracy = 100 * correct / len(test_dataset)
            print(f'Test after epoch {epoch + 1}: loss: {(running_loss / len(test_loader)):.3f}, accuracy: {test_accuracy:.2f}%')

        torch.save(model.state_dict(), weights_path)                
elif Evaluate:
    for resid_levels in range(1, 4):
        model_path = 'models/' + dataset + '/' + str(
            resid_levels) + '_residuals.pt'
        model = CIFAR10Model((32, 32))
        prepare(model, True, resid_levels, resid_levels)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.eval() 
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()
        test_accuracy = 100 * correct / total
        print(f'Test Accuracy: {test_accuracy:.2f}%')
        print(f'loss: {running_loss / len(test_loader):.3f}')
        