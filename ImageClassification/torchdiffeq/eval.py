import os
import argparse
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from model import ODEfunc, ODEBlock, get_mnist_downsampling_layers, get_fc_layers, get_cifar10_downsampling_layers
from utils import accuracy, get_logger, count_parameters
from data import get_mnist_loaders, get_cifar10_loaders

parser = argparse.ArgumentParser()
parser.add_argument('--mnist', type=eval, default=False, choices=[True, False])
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--method', type=str, default='dopri8')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--path', type=str, default='./experiment1')
args = parser.parse_args()

from torchdiffeq import odeint

if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if args.mnist:
        downsampling_layers = get_mnist_downsampling_layers()
    else:
        downsampling_layers = get_cifar10_downsampling_layers()

    feature_layers = [ODEBlock(ODEfunc(64), odeint, args.tol, args.method)]
    fc_layers = get_fc_layers()

    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)

    # Load the saved model
    checkpoint = torch.load(os.path.join(args.path, 'model.pth'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    if args.mnist:
        _, test_loader, _ = get_mnist_loaders(args.batch_size, args.batch_size)
    else:
        _, test_loader, _ = get_cifar10_loaders(args.batch_size, args.batch_size)

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                     record_shapes=True, 
                     profile_memory=True, 
                     with_stack=True, 
                     on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs')) as prof:
            with record_function("model_inference"):
                test_acc = accuracy(model, test_loader, device)
                prof.step()  # Manually step the profiler

    print(f'Test Accuracy: {test_acc:.4f}')
