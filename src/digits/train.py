# -*- coding: utf-8 -*-
"""
Dependances : 
- python (3.8.0)
- numpy (1.19.2)
- torch (1.7.1)
- POT (0.7.0)
- Cuda

command:
python3 train.py
"""


import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# torch.multiprocessing.set_sharing_strategy('file_system')

# import torch.nn.functional as F
from models import Classifier2, weights_init, Cnn_generator

from utils import *
from jumbot import Jumbot
    
import argparse
import wandb
import os

parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')

parser.add_argument("--batchsize", type=int, default=500)
parser.add_argument("--nclass", type=int, default=10)
parser.add_argument("--seed", type=int, default=1980)
parser.add_argument("--n_epochs", type=int, default=100)

parser.add_argument("--eta1", type=float, default=0.1)
parser.add_argument("--eta2", type=float, default=0.1)
parser.add_argument("--tau", type=float, default=1.0)
parser.add_argument("--epsilon", type=float, default=0.1)

parser.add_argument("--wandb_entity", type=str, default='rlopt', help="entitiy of wandb team")
parser.add_argument("--wandb_project_name", type=str, default='default_project', help="entitiy of wandb project")
parser.add_argument('--wandb_offline', action = 'store_true')
args = parser.parse_args()

####### Wandb

# TODO
wandb_exp_name = f'{args.batchsize}_seed_{args.seed}'
if args.wandb_offline:
    os.environ["WANDB_MODE"] = "dryrun"

wandb.init(config=args,
            project=args.wandb_project_name,
            name=wandb_exp_name,
            entity=args.wandb_entity)

print(f'wandb_project_name: f{args.wandb_project_name}')
print(f'wandb_exp_name: f{wandb_exp_name}')


batch_size = args.batchsize
nclass = args.nclass
np.random.seed(args.seed)

# pre-processing to tensor, and mean subtraction

######DATASETS
### TRAIN sets
transform_svhn = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

train_svhn_trainset = datasets.SVHN('./data', split='train', download=True,
                                transform=transform_svhn)

print('nb source data : ', len(train_svhn_trainset))

source_data = torch.zeros((len(train_svhn_trainset), 3, 32, 32))
source_labels = torch.zeros((len(train_svhn_trainset)))

for i, data in enumerate(train_svhn_trainset):
    source_data[i] = data[0]
    source_labels[i] = data[1]

train_batch_sampler = BalancedBatchSampler(source_labels, batch_size=batch_size)
train_svhn_loader = torch.utils.data.DataLoader(train_svhn_trainset, batch_sampler=train_batch_sampler)

transform_mnist = transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

train_mnist_trainset = datasets.MNIST('./data', train=True, download=True,
                                    transform=transform_mnist)
train_mnist_loader = torch.utils.data.DataLoader(train_mnist_trainset, batch_size=batch_size, shuffle=True)


### TEST sets
test_svhn_loader = torch.utils.data.DataLoader(
        datasets.SVHN('./data', split='test', transform=transform_svhn, download=True),
        batch_size=batch_size, shuffle=False)

test_mnist_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transform_mnist),
        batch_size=batch_size, shuffle=False)
    

####### Main
model_g = Cnn_generator().cuda().apply(weights_init)
model_f = Classifier2(nclass=nclass).cuda().apply(weights_init)

model_g.train()
model_f.train()

jumbot = Jumbot(model_g, model_f, n_class=nclass, eta1=args.eta1, eta2=args.eta2, tau=args.tau, epsilon=args.epsilon)
loss = jumbot.source_only(train_svhn_loader)
loss = jumbot.fit(train_svhn_loader, train_mnist_loader, test_mnist_loader, n_epochs=args.n_epochs)

source_acc =jumbot.evaluate(test_svhn_loader)
target_acc =jumbot.evaluate(test_mnist_loader)

# print("source_acc = {}, target_acc ={}".format(source_acc, target_acc))

wandb.log({
            'source_acc': source_acc,
            'target_acc': target_acc,
        })
