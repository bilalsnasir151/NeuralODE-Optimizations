import torch
import torch.nn as nn
import numpy as np
import os
import logging

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def inf_generator(iterable):

    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()
''' INF GENERATOR FUNCTION

** 
Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
**

basically feeds in batches of data in the training loop
'''

def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates, lr):
    initial_learning_rate = lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn
''' LEARNING RATE WITH DECAY
** dynamically adjusts learning rate based on number of iterations completed **

batch_size = the specified batch size from the args
batch_denom = normalization factor that scales the learning rate according to the batch size used
batches_per_epoch = number of batches that are processed in a single epoch
boundary_epochs = a list of what epochs to start adjusting learning rates
decay_rates = influences how the learning rate changes over different epoch boundaries

** CODE EXPLANATION **


'''

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)
''' ONE HOT EXPLANATION

basically allows for classification

x represents array of integer labels (10 classes, 10 labels), basically what class it is 
K represents total number of classes, output vector has k components, one component marked as 1 and rest as 0.

First reshape x by adding a new axis. For example, if x is [0, 2, 1], it becomes [[0], [2], [1]]
Then create 1d array of integers from 0 to k-1. For K = 3, this would result in [[0, 1, 2]]
Now element wise comparison, results in 2d boolean array:

For example, if x = [0, 2, 1] and K = 3, the comparison would be
x = [
    [0],       == [[0, 1, 2]]
    [2],
    [1]
]
results in
[
    [ True, False, False],
    [False, False,  True],
    [False,  True, False]
]

'''

def accuracy(model, dataset_loader, device, profiler=None):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
        
        if profiler:
            profiler.step()  # Step the profiler during inference
    return total_correct / len(dataset_loader.dataset)
''' ACCURACY EXPLANATION


'''

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger
