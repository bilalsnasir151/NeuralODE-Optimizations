import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=128, test_batch_size=1000):

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=1, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=1, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=1, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader
''' DATASET LOADER for MNIST


** TASK IS TO LOAD MNIST DATA **

- batch size = number of samples in each batch for the training dataset
- test_batch_size = number of sample in each batch for testing and training eval datasets

CODE EXPLANATION

transform_train & transform_test:
    .ToTensor(): converts a PIL image into a PyTorch tensor. Scales image data to a range of 0.0-1.0 by dividing all pixel values by 255.
        TLDR: normalizes pil image in a pytorch tensor
    .Compose(): will apply following transformations to the data

DataLoader(): 
    root = MNIST: Looks for mnist dataset
    train = True/False: says whether to look at the training portion or testing portion of the dataset
    download = True: says to download dataset
    transform: specifies what transformations (defined in transform_train and transform_test)
    batch_size: number of data points to load in each batch of training. impacts speed and stability. 
    shuffle: asks the loader to shuffle the order of dataset. allows for training of different batches per epoch. results in a more generalized model
    num_workers: specifies how many subprocesses to use for data loading. more workers increases throughput and decrese time it takes to load data at the cost of increased memory usage. 
        TLDR: specifies how many cpus to use for data loading, more = more cpu usage, memory usage.
    drop_last: drops last incomplete batch so that it doesnt train on a batch thats less than batch_size

    RETURNS three loader configurations
'''
