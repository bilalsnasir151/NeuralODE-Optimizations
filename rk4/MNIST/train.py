import os
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
from torchdyn.core import NeuralODE
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--save_path', type=str, default='./model.pth')
args = parser.parse_args()

if __name__ == '__main__':

    ## FIND GPU IF AVAILABLE ##
    # CUDA First Option #
    # MPS Second Option #
    # CPU Last Resort # 
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    ## Load Data ##
    # 1. Load MNIST Data
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    print("TRAIN: ", train_dataset)
    print("data: ", train_dataset.data.size())
    print("targets: ", train_dataset.targets.size())

    print("TEST: ", test_dataset)

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # figure = plt.figure(figsize=(10, 8))
    # cols, rows = 5, 5
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    #     img, label = train_dataset[sample_idx]
    #     figure.add_subplot(rows, cols, i)
    #     plt.title(label)
    #     plt.axis("off")
    #     plt.imshow(img.squeeze(), cmap="gray")
    # plt.show()

    ## CONSTRUCT THE NETWORK ##
    # NETWORK STRUCTURE #


    # 1. Downsample the data
    class Downsample(nn.Module):
        def __init__(self):
            super(Downsample, self).__init__()
            self.downsample = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64,64,kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64,64,kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.downsample(x)

    # 2. ODE function
    
    class ConcatConv2d(nn.Module):

        def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
            super(ConcatConv2d, self).__init__()
            module = nn.ConvTranspose2d if transpose else nn.Conv2d
            self._layer = module(
                dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias
            )

        def forward(self, t, x):
            tt = torch.ones_like(x[:, :1, :, :]) * t
            ttx = torch.cat([tt, x], 1)
            return self._layer(ttx)
    
    class ODEfunc(nn.Module):

        def __init__(self, dim):
            super(ODEfunc, self).__init__()
            self.norm1 = nn.GroupNorm(min(32,dim), dim)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
            self.norm2 = nn.GroupNorm(min(32,dim), dim)
            self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
            self.norm3 = nn.GroupNorm(min(32,dim), dim)

        def forward(self, t, x):
            out = self.norm1(x)
            out = self.relu(out)
            out = self.conv1(t, out)

            out = self.norm2(out)
            out = self.relu(out)
            out = self.conv2(t, out)

            out = self.norm3(out)
            return out
        
    class NeuralODE(nn.Module):

        def __init__(self, odefunc, odeint, tol, method):
            super(NeuralODE, self).__init__()
            self.odefunc = odefunc
            self.integration_time = torch.tensor([0, 1]).float()
            self.odeint = odeint
            self.tol = tol
            self.method = method

        def forward(self, x):
            self.integration_time = self.integration_time.type_as(x)
            rtol = torch.as_tensor(self.tol, dtype=torch.float32, device=x.device)
            atol = torch.as_tensor(self.tol, dtype=torch.float32, device=x.device)
            out = self.odeint(self.odefunc, x, self.integration_time, rtol=rtol, atol=atol, method=self.method)
            print("Output of odeint:", out)  # Debugging line to inspect the output
            return out


    # 3. Output prediction / classifier
    class Classifier(nn.Module):
        def __init__(self, downsample, neuralODE):
            super(Classifier, self).__init__()
            self.downsample = downsample
            self.neuralODE = neuralODE
            self.classifier = nn.Sequential(  
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 10)
            )

        def forward(self, x):
            x = self.downsample(x)
            x, _ = self.neuralODE(x)  # Assuming the NeuralODE returns a tuple (output, info)
            x = self.classifier(x)
            return x


    ## Training Loop ##
    # 1. Train Model on MNIST Data
    # 2. Output Training accuracy
    model = Classifier(Downsample(), NeuralODE(ODEfunc(64), odeint=odeint, tol=args.tol, method='rk4')).to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.nepochs):
        model.train()
        for inputs, labels in train_loader:
            print("batch starterd")
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print("batch done")

        # Optionally add validation accuracy check
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch + 1}/{args.nepochs}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')

            # Save the model
    torch.save(model.state_dict(), args.save_path)
    print(f'Model saved to {args.save_path}')