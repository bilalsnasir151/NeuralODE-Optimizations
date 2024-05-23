import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from model import ODEfunc, ODEBlock, get_mnist_downsampling_layers, get_fc_layers, get_cifar10_downsampling_layers
from utils import accuracy, get_logger, one_hot
from data import get_mnist_loaders, get_cifar10_loaders
import torch.profiler

parser = argparse.ArgumentParser()
parser.add_argument('--mnist', type=eval, default=False, choices=[True, False])
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--save', type=str, default='./evaluation')
parser.add_argument('--profiler', action='store_true', help='Enable profiling')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

if __name__ == '__main__':

    # Create save directory
    os.makedirs(args.save, exist_ok=True)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    # Load the model
    if args.mnist:
        downsampling_layers = get_mnist_downsampling_layers()
    else:
        downsampling_layers = get_cifar10_downsampling_layers()

    feature_layers = [ODEBlock(ODEfunc(64), odeint, 1e-3)]  # Use a default tolerance
    fc_layers = get_fc_layers()
    
    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Load the dataset
    if args.mnist:
        _, test_loader, _ = get_mnist_loaders(128, args.batch_size)  # We only need the test loader
    else:
        _, test_loader, _ = get_cifar10_loaders(128, args.batch_size)

    logger.info(model)
    logger.info('Number of parameters: {}'.format(sum(p.numel() for p in model.parameters())))

    # Profiling
    if args.profiler:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.save),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            test_acc = accuracy(model, test_loader, device, prof)
            logger.info('Test Accuracy: {:.4f}'.format(test_acc))
        
        # Print profiler results
        print(prof.key_averages().table(sort_by="cpu_time_total"))
        print(f"Profiling data saved to {args.save}")
    else:
        test_acc = accuracy(model, test_loader, device)
        logger.info('Test Accuracy: {:.4f}'.format(test_acc))

    print(f"Test Accuracy: {test_acc:.4f}")
