import torch
import torch.nn as nn
from ImageClassification.torchdiffeq.utils import norm, Flatten
import torchode as to

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
        # Broadcast t to match the shape of x (batch, 1, height, width)
        tt = t.view(-1, 1, 1, 1).expand_as(x[:, :1, :, :])
        ttx = torch.cat([tt, x], dim=1)
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

    def forward(self, t, y, args):
        self.nfe += 1

        # Unpack the shape information
        channels, height, width = args

        # Reshape y from 2D to 4D for convolution
        print(f"Shape before reshaping: {y.shape}")
        y = y.view(-1, channels, height, width)
        print(f"Shape after reshaping: {y.shape}")

        out = self.norm1(y)
        out = self.relu(out)
        out = self.conv1(t, out)
        print(f"Shape after conv1: {out.shape}")

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        print(f"Shape after conv2: {out.shape}")

        out = self.norm3(out)

        # Flatten back to 2D before returning
        out = out.flatten(start_dim=1)
        print(f"Shape after flattening: {out.shape}")

        return out


class ODEBlock(nn.Module):
    def __init__(self, odefunc, tol=1e-7, method='dopri5'):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.tol = tol
        self.method = method

        # Initialize TorchODE components
        self.term = to.ODETerm(odefunc, with_args=True)
        self.step_method = to.Dopri5(term=self.term)
        self.step_size_controller = to.IntegralController(atol=tol, rtol=tol, term=self.term)
        self.adjoint = to.AutoDiffAdjoint(self.step_method, self.step_size_controller)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Flatten the input tensor to 2D (batch_size, channels*height*width)
        y0 = x.view(batch_size, -1)
        print(f"Shape of y0: {y0.shape}")  # Expect [128, channels*height*width]

        # Time evaluation: Properly batch the time points
        t_eval = torch.tensor([0.0, 1.0], device=x.device).unsqueeze(0).repeat(batch_size, 1)
        print(f"Shape of t_eval: {t_eval.shape}")  # Should be [batch_size, 2]

        # Create the initial value problem
        problem = to.InitialValueProblem(y0=y0, t_eval=t_eval)
        print(f"Shape of y0 after creating problem: {problem.y0.shape}")  # Should match y0 shape

        # Solve the problem
        solution = self.adjoint.solve(problem, args=(channels, height, width))
        print("THIS IS HOW SOLUTION LOOKS: ", solution)
        print("SOLUTIONS SHAPE: ", solution.ys)

        # Select the last time step's output for reshaping
        solution_ys = solution.ys[-1]
        print(f"Shape of solution.ys[-1]: {solution_ys.shape}")  # Should match [batch_size, -1]

        # Reshape back to the original 4D shape (batch_size, channels, height, width)
        try:
            out = solution_ys.view(batch_size, channels, height, width)
        except RuntimeError as e:
            print(f"Error reshaping: {e}")
            print(f"Expected shape: {[batch_size, channels, height, width]}")
            print(f"Actual number of elements: {solution_ys.numel()}")
            raise

        return out
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
