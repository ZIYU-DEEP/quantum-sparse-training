"""
[Title] experiment-hessian.py
[Usage] This is a temporary file to calculate the prediction entropy.
"""
print("Here")
from mpi4py import MPI
num_gpus_per_node = 4
comm = MPI.COMM_WORLD
num_gpu_ranks = comm.Get_size()
rank = comm.Get_rank()
gpu = rank%num_gpus_per_node
print('rank {} using gpu {}'.format(rank, gpu))

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu)

from helper import utils, pruner, hessian
from loader import load_dataset
from optim import Model
from pathlib import Path
import numpy as np
from torch import nn
import torch
import logging
import torch.nn
import argparse


# ##################################################################
# 0. Set the arguments
# ##################################################################
parser = argparse.ArgumentParser()

parser.add_argument('-pt', '--path', type=str, default='./final_path',
                    help='The path to get results.')
parser.add_argument('-pi', '--prune_indicator', type=int, default=1,
                    help='1 if prune, else 0.')
parser.add_argument('-pr', '--prune_ratio', type=float, default=0.9,
                    help='The ratio for sparse training.')
parser.add_argument('-no', '--no', type=int, default=None,
                    help='The epoch number to be loaded from the state_dicts folder.')

# Better to leave them as default
parser.add_argument('-ul', '--use_loader', type=str, default='test',
                    help='The loader to use in evaluating the fisher.',
                    choices=['train', 'test', 'clean', 'noisy'])
parser.add_argument('-dv', '--device', type=str, default='cuda',
                    help='Choose from cpu, cuda, and tpu.')
parser.add_argument('-bs', '--batch_size', type=int, default=128,
                    help='The batch size for training.')
parser.add_argument('-ipd', '--init_poly_deg', type=int, default=64,
                    help='The iterations used to compute spectrum range.')
parser.add_argument('-pd', '--poly_deg', type=int, default=256,
                    help='The higher the parameter the better the approximation.')

# Arguments for setting network
parser.add_argument('-nt', '--net_name', type=str, default='mlp',
                    help='The name for your network',
                    choices=['mlp', 'alexnet', 'preresnet', 'resnet',
                             'densenet', 'vgg11', 'vgg11_bn', 'vgg13',
                             'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19',
                             'vgg19_bn', 'mnist_lenet', 'mnist_alexnet'])
parser.add_argument('-in', '--in_dim', type=int, default=12,
                    help='The feature dimension for the input data X.')
parser.add_argument('-ot', '--out_dim', type=int, default=2,
                    help='The number of classes of the output data y.')
parser.add_argument('-ha', '--hidden_act', type=str, default='tanh',
                    help='The activation for hidden layers, e.g., tanh, relu, softmax, sigmoid.')
parser.add_argument('-oa', '--out_act', type=str, default='softmax',
                    help='The activation for the output layer, e.g., softmax, sigmoid.')
parser.add_argument('-hd', '--hidden_dims', type=str, default='10-7-5-4-3',
                    help='The hidden dimensions for MLP; using hypen to connect numbers.')
parser.add_argument('-dp', '--depth', type=int, default=32,
                    help='The depth for preresnet or densenet.')
parser.add_argument('-wf', '--widen_factor', type=int, default=4,
                    help='The widen factor for resnet.')
parser.add_argument('-dr', '--dropRate', type=int, default=0,
                    help='The drop rate for dense net.')
parser.add_argument('-gr', '--growthRate', type=int, default=12,
                    help='The growth rate for dense net.')
parser.add_argument('-cr', '--compressionRate', type=int, default=1,
                    help='The compression rate for densenet.')

# Arguments for loading datasets
parser.add_argument('-ln', '--loader_name', type=str, default='toy',
                    help='The name for your dataset.',
                    choices=['toy', 'mnist', 'cifar10', 'cifar100',
                             'fmnist', 'cifar100_tpu',
                             'cifar10_noisy', 'fmnist_noisy', 'tiny_imagenet', 'imagenet'])
parser.add_argument('-rt', '--root', type=str, default='./data/',
                    help='The root path for stored data.')
parser.add_argument('-fn', '--filename', type=str, default='toy',
                    help='The filename for your data, e.g., toy, MNIST or CIFAR10.')
parser.add_argument('-ts', '--test_size', type=float, default=0.2,
                    help='The test split ratio for the toy dataset.')
parser.add_argument('-rs', '--random_state', type=int, default=42,
                    help='Use 0 to let the machine set a random random seed for you.')
parser.add_argument('-dl', '--download', type=int, default=0,
                    help='1 if you need download the dataset, otherwise 0.')

p = parser.parse_args()


# ##################################################################
# 0. Define Global Variables
# ##################################################################
final_path = Path(p.path)
# prune_indicator = p.prune_indicator
prune_indicator = 0
batch_size = p.batch_size
# device = p.device
# device = 'cuda:{}'.format(gpu)
device = 'cuda'
# epochs = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,55,60,65,70,75,80,85,90,95,100,110,120,130,140,150,160,170,180,190,200,210,240]
# epochs = [0, 2, 4, 6]
# if p.no != None:
#     epoch = p.no
# else:
#     epoch = epochs[rank]
# step = [0,2,4,6,8,10,12,14,16,18,20,25,30,35,40,45,50][rank]
# step=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100][rank]
step=np.arange(0,204,4)[rank]
print('Rank {} using gpu {} for epoch {}.'.format(rank, device, step))
state_dict_path = final_path / 'state_dicts' / f'reupload_step_{step}.pkl'
log_path = final_path / 'hessian_spectrum.log'
prune_ratio = p.prune_ratio  # Just a placeholder


# ##################################################################
# 1. Prepartions
# ##################################################################
# Set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('log epoch {}'.format(step))
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info(state_dict_path)

# Set neccessities
device = torch.device(device)
criterion = nn.CrossEntropyLoss()

# Set the dataset
dataset = load_dataset(p.loader_name,
                       p.root,
                       p.filename,
                       p.test_size)

train_loader, test_loader, _ = dataset.loaders(batch_size=p.batch_size, num_workers=32)

# test_data = Subset(dataset, torch.arange(0, 1280))
# test_loader = DataLoader(test_data, batch_size=128)

# Create the function to set the network
def set_network(prune_indicator,
                prune_ratio,
                state_dict_path,
                device):
    """
    Set a network.
    """
    # Load the dict
    state_dict = torch.load(state_dict_path, map_location=device)

    # Set the network
    # net = ResNet(out_dim=100).to(device=device)
    model = Model()
    model.set_network(p.net_name, p.in_dim, p.out_dim, p.hidden_act,
                  p.out_act, p.hidden_dims, p.depth, p.widen_factor,
                  p.dropRate, p.growthRate, p.compressionRate)

    net = model.net
    net = torch.nn.DataParallel(net).to(device)

    # Prune the network if needed
    if prune_indicator:
        print('pruned')
        try:
            pruner.global_prune(net, 'l1', prune_ratio, False)
            net = utils.load_state_dict_(net, state_dict)
        except:
            pruner.global_prune(net, 'l1', prune_ratio, True)
            net = utils.load_state_dict_(net, state_dict)
    else:
        net = utils.load_state_dict_(net, state_dict)

    # Load the dict to net
    return net


# ##################################################################
# 1. Calculate the Hessian
# ##################################################################
# Set network
net = set_network(prune_indicator, prune_ratio, state_dict_path, device)

# Set the loader
if p.use_loader == 'train':
    data_loader = train_loader
elif p.use_loader == 'test':
    data_loader = test_loader
else:
    data_loader = train_loader

# Get logging
logger.info(f'Getting hessian for batch size as {batch_size}...')

# Wrap the Hessian class
H = hessian.Hessian(loader=data_loader,
                    model=net,
                    hessian_type='Hessian')

# Get the Hessian eigenvalue and the associated density
H_eigval, H_eigval_density = H.LanczosApproxSpec(init_poly_deg=p.init_poly_deg, poly_deg=p.poly_deg, poly_points=256)

# Save the eigenvalue and the corresponding density
np.save(final_path / 'hessian_eigval_epoch_{}'.format(step), H_eigval)
np.save(final_path / 'hessian_eigval_density_epoch_{}'.format(step), H_eigval_density)
logger.info(f'The spectrum is now saved in npz files.')


# Log the results
logger.info('Done!')
