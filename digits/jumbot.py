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
import torch.utils.data
import itertools
import torch.nn.functional as F

import ot

from utils import model_eval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Jumbot(object):
    """Jumbot class"""
    def __init__(self, model_g, model_f, n_class, eta1=0.001, eta2=0.0001, tau=1., epsilon=0.1):
        """
        Initialize jumbot method.
        args :
        - model_g : feature exctrator (torch.nn)
        - model_f : classification layer (torch.nn)
        - n_class : number of classes (int)
        - eta_1 : feature comparison coefficient (float)
        - eta_2 : label comparison coefficient (float)
        - tau : marginal coeffidient (float)
        - epsilon : entropic regularization (float)
        """
        self.model_g = model_g   # target model
        self.model_f = model_f
        self.n_class = n_class
        self.eta1 = eta1  # weight for the alpha term
        self.eta2 = eta2 # weight for target classification
        self.tau = tau
        self.epsilon = epsilon
        print('eta1, eta2, tau, epsilon: ', self.eta1, self.eta2, self.tau, self.epsilon)
    
    def fit(self, source_loader, target_loader, test_loader, n_epochs, criterion=nn.CrossEntropyLoss()):
        """
        Run jumbot method.
        args :
        - source_loader : source dataset 
        - target_loader : target dataset
        - test_loader : test dataset
        - n_epochs : number of epochs (int)
        - criterion : source loss (nn)
        
        return:
        - trained model
        """
        target_loader_cycle = itertools.cycle(target_loader)
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=2e-4)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=2e-4)

        for id_epoch in range(n_epochs):
            self.model_g.train()
            self.model_f.train()
            for i, data in enumerate(source_loader):
                ### Load data
                xs_mb, ys = data
                xs_mb, ys = xs_mb.cuda(), ys.cuda()
                xt_mb, _ = next(target_loader_cycle)
                xt_mb = xt_mb.cuda()
                
                g_xs_mb = self.model_g(xs_mb.cuda())
                f_g_xs_mb = self.model_f(g_xs_mb)
                g_xt_mb = self.model_g(xt_mb.cuda())
                f_g_xt_mb = self.model_f(g_xt_mb)
                pred_xt = F.softmax(f_g_xt_mb, 1)

                ### loss
                s_loss = criterion(f_g_xs_mb, ys.cuda())

                ###  Ground cost
                embed_cost = torch.cdist(g_xs_mb, g_xt_mb)**2
                
                ys = F.one_hot(ys, num_classes=self.n_class).float()
                t_cost = - torch.mm(ys, torch.transpose(torch.log(pred_xt), 0, 1))
                
                total_cost = self.eta1 * embed_cost + self.eta2 * t_cost

                #OT computation
                a, b = ot.unif(g_xs_mb.size()[0]), ot.unif(g_xt_mb.size()[0])
                pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, total_cost.detach().cpu().numpy(),
                                                             self.epsilon, self.tau)
                # To get DeepJDOT (https://arxiv.org/abs/1803.10081) comment the line above 
                # and uncomment the following line:
                #pi = ot.emd(a, b, total_cost.detach().cpu().numpy())
                pi = torch.from_numpy(pi).float().cuda()

                # train the model 
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                da_loss = torch.sum(pi * total_cost)
                tot_loss = s_loss + da_loss
                tot_loss.backward()

                optimizer_g.step()
                optimizer_f.step()
            
            print('epoch, loss : ', id_epoch, s_loss.item(), da_loss.item())
            if id_epoch%10 == 0:
                source_acc = self.evaluate(source_loader)
                target_acc = self.evaluate(test_loader)
                print('source and test accuracies : ', source_acc, target_acc)
        
        return tot_loss

    def source_only(self, source_loader, criterion=nn.CrossEntropyLoss()):
        """
        Run source only.
        args :
        - source_loader : source dataset 
        - criterion : source loss (nn)
        
        return:
        - trained model
        """
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=2e-4)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=2e-4)

        for id_epoch in range(10):
            self.model_g.train()
            self.model_f.train()
            for i, data in enumerate(source_loader):
                ### Load data
                xs_mb, ys = data
                xs_mb, ys = xs_mb.cuda(), ys.cuda()
                
                g_xs_mb = self.model_g(xs_mb.cuda())
                f_g_xs_mb = self.model_f(g_xs_mb)

                ### loss
                s_loss = criterion(f_g_xs_mb, ys.cuda())

                # train the model 
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                tot_loss = s_loss
                tot_loss.backward()

                optimizer_g.step()
                optimizer_f.step()
        
        return tot_loss
    

    def evaluate(self, data_loader):
        score = model_eval(data_loader, self.model_g, self.model_f)
        return score
