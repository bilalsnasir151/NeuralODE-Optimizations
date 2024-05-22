import os
import argparse
import torch
import torch.nn as nn
from ImageClassification.model import ODEfunc, ODEBlock, get_fc_layers, get_cifar10_downsampling_layers, get_mnist_downsampling_layers
from ImageClassification.utils import accuracy, get_logger
from ImageClassification.data import get_cifar10_loaders, get_mnist_loaders

parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--mnist', type=eval, default=False, choices=[True, False])
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

if __name__ == '__main__':
    logger = get_logger(logpath='./eval_logs', filepath=os.path.abspath(__file__))
    logger.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Build the model architecture
    if args.mnist:
        downsampling_layers = get_mnist_downsampling_layers()
    else:
        downsampling_layers = get_cifar10_downsampling_layers()
    
    feature_layers = [ODEBlock(ODEfunc(64), odeint, args.tol)]
    fc_layers = get_fc_layers()
    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)

    # Load the pre-trained model
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    logger.info('Model loaded from {}'.format(args.model_path))

    # Set the model to evaluation mode
    model.eval()

    if args.mnist:
        _, test_loader, _ = get_mnist_loaders(batch_size=args.batch_size, test_batch_size=args.batch_size)
    else:
        # Get the CIFAR-10 data loaders
        _, test_loader, _ = get_cifar10_loaders(batch_size=args.batch_size, test_batch_size=args.batch_size)

    # Evaluate the model
    with torch.no_grad():
        test_acc = accuracy(model, test_loader, device)
    
    logger.info('Test Accuracy: {:.4f}'.format(test_acc))
    print('Test Accuracy: {:.4f}'.format(test_acc))
