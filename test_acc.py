"""
[Title] experiment-hessian.py
[Usage] This is a temporary file to calculate the prediction entropy.
"""

from mpi4py import MPI
num_gpus_per_node = 4
comm = MPI.COMM_WORLD
num_gpu_ranks = comm.Get_size()
rank = comm.Get_rank()
gpu = rank%num_gpus_per_node
print('rank {} using gpu {}'.format(rank, gpu))

from helper import utils, pruner, hessian
from pathlib import Path
from torch import nn
from PIL import Image
from functools import reduce
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from loader.loader_cifar10 import CIFAR10Loader
from loader.loader_cifar100 import CIFAR100Loader
from network.res_net import ResNet
from collections import OrderedDict

import math
import time
import torch
import joblib
import logging
import torch.nn
import argparse
import numpy as np
import seaborn as sea
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn.utils.prune as torch_prune
import torchvision.transforms as transforms


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
parser.add_argument('-no', '--no', type=int, default=0,
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

p = parser.parse_args()


# ##################################################################
# 0. Define Global Variables
# ##################################################################
final_path = Path(p.path)
prune_indicator = p.prune_indicator
batch_size = p.batch_size
# device = p.device
device = torch.device('cuda:{}'.format(gpu))
# device = torch.device('cpu')
# state_dict_path = final_path / 'state_dicts' / f'epoch_{p.no}.pkl'

log_path = final_path / 'hessian_spectrum.log'
prune_ratio = p.prune_ratio  # Just a placeholder

# Set neccessities
device = torch.device(device)
criterion = nn.CrossEntropyLoss()

# Set the dataset
dataset = CIFAR100Loader()
train_loader, test_loader, _ = dataset.loaders(batch_size=batch_size,
                                            shuffle_train=False,
                                            shuffle_test=False)

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
    net = ResNet(out_dim=100).to(device=device)

    # Prune the network if needed
    if prune_indicator:
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

def test(net, test_loader):
    criterion = nn.CrossEntropyLoss()

    # Set up relavant information
    epoch_loss, correct, n_batches = 0.0, 0, 0

    # Start testing
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            # Get data
            inputs, y, _ = data
            inputs, y = inputs.to(device), y.to(device)
            outputs = net(inputs)

            # Calculate loss and accuracy
            loss = criterion(outputs, y)

            # Record loss
            epoch_loss += loss.item()
            correct += (outputs.argmax(1) == y).type(torch.float).sum().item()
            n_batches += 1

    # Log results
    print('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
    print('Test Accuracy: {:.3f}'.format(correct / len(test_loader.dataset)))

# Set the loader
if p.use_loader == 'train':
    data_loader = train_loader
elif p.use_loader == 'test':
    data_loader = test_loader
else:
    data_loader = train_loader

# ##################################################################
# 1. Calculate the Hessian
# ##################################################################
# Set network
if rank == 0:
    for step in range(22, 96):

        print('step ', step)
        state_dict_path = final_path / 'state_dicts' / f'step_{step}.pkl'
        net = set_network(prune_indicator, prune_ratio, state_dict_path, device)

        if rank == 0:
            # Wrap the Hessian class
            test(net, data_loader)