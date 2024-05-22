import os
import argparse
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
from model import ODEfunc, ODEBlock, get_mnist_downsampling_layers, get_fc_layers, get_cifar10_downsampling_layers
from utils import RunningAverageMeter, inf_generator, learning_rate_with_decay, accuracy, makedirs, get_logger, count_parameters
from data import get_mnist_loaders, get_cifar10_loaders

parser = argparse.ArgumentParser()
parser.add_argument('--mnist', type=eval, default=False, choices=[True, False])
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

if __name__ == '__main__':

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)
    
    torch.set_default_dtype(torch.float32)

    #find device
    #device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    #first section of network, primarily focusing on reducing spatial dimensions of input while extracting basic to complex features
    if args.mnist:
        downsampling_layers = get_mnist_downsampling_layers()
    else:
        downsampling_layers = get_cifar10_downsampling_layers()

    feature_layers = [ODEBlock(ODEfunc(64), odeint, args.tol)]
    fc_layers = get_fc_layers()

    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)

    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss().to(device)



    if args.mnist:
        train_loader, test_loader, train_eval_loader = get_mnist_loaders(args.batch_size, args.test_batch_size)
    else:
        train_loader, test_loader, train_eval_loader = get_cifar10_loaders(args.batch_size, args.test_batch_size)

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001], lr = args.lr
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    for itr in range(args.nepochs * batches_per_epoch):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        x, y = data_gen.__next__()
   	x = x.to(device)
    	y = y.to(device)

    	optimizer.zero_grad()

    	# Start a new profiler instance each iteration
    	with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, 
                                           torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else torch.profiler.ProfilerActivity.CPU],
                                record_shapes=True) as prof:
        	with torch.profiler.record_function("model_forward_backward"):
            		logits = model(x)
            		loss = criterion(logits, y)
            		loss.backward()
        	optimizer.step()  # Now inside the profiling context

    	# Profiling results
    	print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    	nfe_forward = feature_layers[0].nfe
    	feature_layers[0].nfe = 0
    	nfe_backward = feature_layers[0].nfe
    	feature_layers[0].nfe = 0

    	batch_time_meter.update(time.time() - end)
    	f_nfe_meter.update(nfe_forward)
    	b_nfe_meter.update(nfe_backward)
    	end = time.time()

    	if itr % batches_per_epoch == 0:
            with torch.no_grad():
            	train_acc = accuracy(model, train_eval_loader, device)
            	val_acc = accuracy(model, test_loader, device)
            	if val_acc > best_acc:
            		torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                	best_acc = val_acc
            	logger.info(
                "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                "Train Acc {:.4f} | Test Acc {:.4f}".format(
                    itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                    b_nfe_meter.avg, train_acc, val_acc
                )
            )
