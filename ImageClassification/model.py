import torch
import torch.nn as nn
from utils import norm, Flatten



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

##Mnist downsampling##
def get_mnist_downsampling_layers():
    downsampling_layers = [
        #applies filters to extract low level features
        nn.Conv2d(1, 64, 3, 1),
        #normalizes values across certain dimensions of input data
        norm(64),
        #activation function
        nn.ReLU(inplace=True),

        nn.Conv2d(64, 64, 4, 2, 1),
        norm(64),
        nn.ReLU(inplace=True),

        nn.Conv2d(64, 64, 4, 2, 1),
    ]        
    return downsampling_layers

##CIFAR 10 downsampling
def get_cifar10_downsampling_layers():
    downsampling_layers = [
        #applies filters to extract low level features
        nn.Conv2d(3, 64, 3, 1, 1),
        #normalizes values across certain dimensions of input data
        norm(64),
        #activation function
        nn.ReLU(inplace=True),

        nn.Conv2d(64, 64, 4, 2, 1),
        norm(64),
        nn.ReLU(inplace=True),

        nn.Conv2d(64, 64, 4, 2, 1),
        norm(64),
        nn.ReLU(inplace=True),
    ]        
    return downsampling_layers

##MNIST NODE BLOCK##
#neural ode concatenation of time
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
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1

        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)

        out = self.norm3(out)
        return out
    
class ODEBlock(nn.Module):

    def __init__(self, odefunc, odeint, tol, method):
        super(ODEBlock, self).__init__()
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
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


##MNIST FC LAYERS##
def get_fc_layers():
    fc_layers = [
    norm(64), 
    nn.ReLU(inplace=True), 
    nn.AdaptiveAvgPool2d((1, 1)), 
    Flatten(), 
    nn.Linear(64, 10)]

    return fc_layers
