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


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
    
class Classifier2(nn.Module):
    ''' Classifier class'''
    def __init__(self, nclass=None):
        super(Classifier2, self).__init__()
        assert nclass!=None
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc2(x)
        return x
    
    
def weights_init(m):
    ''' Weight init function for layers '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)
        
        
def call_bn(bn, x):
    ''' call batch norm layer '''
    return bn(x)


class Cnn_generator(nn.Module):
    '''9 layer CNN feature extractor class'''
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, momentum=0.1):
        self.momentum = momentum 
        super(Cnn_generator, self).__init__()
        self.c1=nn.Conv2d(input_channel, 32,kernel_size=3, stride=1, padding=1)        
        self.c2=nn.Conv2d(32,32,kernel_size=3, stride=1, padding=1)        
        self.c3=nn.Conv2d(32,64,kernel_size=3, stride=1, padding=1)        
        self.c4=nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1)        
        self.c5=nn.Conv2d(64,128,kernel_size=3, stride=1, padding=1)        
        self.c6=nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1)        
        self.linear1=nn.Linear(128*4*4, 128)
        self.bn1=nn.BatchNorm2d(32)
        self.bn2=nn.BatchNorm2d(32)
        self.bn3=nn.BatchNorm2d(64)
        self.bn4=nn.BatchNorm2d(64)
        self.bn5=nn.BatchNorm2d(128)
        self.bn6=nn.BatchNorm2d(128)
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x):
        h=x
        h=self.c1(h)
        h=F.relu(call_bn(self.bn1, h))
        h=self.c2(h)
        h=F.relu(call_bn(self.bn2, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c3(h)
        h=F.relu(call_bn(self.bn3, h))
        h=self.c4(h)
        h=F.relu(call_bn(self.bn4, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c5(h)
        h=F.relu(call_bn(self.bn5, h))
        h=self.c6(h)
        h=F.relu(call_bn(self.bn6, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h = h.view(h.size(0), -1)
        logit=torch.sigmoid(self.linear1(h))
        return logit