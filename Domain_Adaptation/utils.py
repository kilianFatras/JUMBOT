is_torchvision_installed = True
try:
    import torchvision
except:
    is_torchvision_installed = False

import torch
import torch.nn.functional as F
import torch.utils.data
import random
import numpy as np

from torch.utils.data.sampler import BatchSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#-------- GEOMLOSS EUCLIDEAN DISTANCE --------
class Sqrt0(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        result = input.sqrt()
        result[input < 0] = 0
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        grad_input = grad_output / (2*result)
        grad_input[result == 0] = 0
        return grad_input

def sqrt_0(x):
    return Sqrt0.apply(x)


def squared_distances(x, y):
    '''
        Compute the squared eculidean matrix 
        Adapted from Geomloss
    '''
    if x.dim() == 2:
        D_xx = (x*x).sum(-1).unsqueeze(1)  # (N,1)
        D_xy = torch.matmul( x, y.permute(1,0) )  # (N,D) @ (D,M) = (N,M)
        D_yy = (y*y).sum(-1).unsqueeze(0)  # (1,M)
    elif x.dim() == 3:  # Batch computation
        D_xx = (x*x).sum(-1).unsqueeze(2)  # (B,N,1)
        D_xy = torch.matmul( x, y.permute(0,2,1) )  # (B,N,D) @ (B,D,M) = (B,N,M)
        D_yy = (y*y).sum(-1).unsqueeze(1)  # (B,1,M)
    else:
        print("x.shape : ", x.shape)
        raise ValueError("Incorrect number of dimensions")

    return D_xx - 2*D_xy + D_yy


def distances(x, y):
    '''apply sqrt element wise'''
    return sqrt_0( squared_distances(x,y) )


def SCE(proba1, proba2, eta_1=0.01, eta_2=1):
    '''
        Compute the symmetric cross entropy

        args : 
        - proba1 : probability vector batches (one hot vectors)
        - proba2 : probability vector batches (one hot vectors)

        Return :
        Symmetric cross entropy
        
    '''
    
    proba1_clamp = torch.clamp(proba1, 1e-7, 1)
    proba2_clamp = torch.clamp(proba2, 1e-7, 1)
    sce = - eta_1 * torch.mm(proba1, torch.transpose(torch.log(proba2_clamp), 0, 1))
    sce += - eta_2 * torch.mm(torch.log(proba1_clamp), torch.transpose(proba2, 0, 1))
    return sce


def CE(proba1, proba2):
    '''
        Compute the symmetric cross entropy

        args : 
        - proba1 : probability vector batches (one hot vectors)
        - proba2 : probability vector batches (one hot vectors)

        Return :
        Symmetric cross entropy
        
    '''
    
    proba2_clamp = torch.clamp(proba2, 1e-7, 1)
    sce = - torch.mm(proba1, torch.transpose(torch.log(proba2_clamp), 0, 1))
    return sce

def FOT(class_a, class_b):
    '''
        compute matched labels matrix 
    '''
    
    same_cl = torch.ne(class_a[:,None], class_b[None,:]).float()
    same_cl[same_cl>0] *= 10000000
    return same_cl


#-------- Eval function --------

def model_eval(dataloader, model_g, model_f):
    model_g.eval()
    model_f.eval()
    total_samples =0
    correct_prediction = 0
    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device)
            label = label.long().to(device)
            gen_output = model_g(img)
            pred = F.softmax(model_f(gen_output), 1)
            correct_prediction += torch.sum(torch.argmax(pred,1)==label)
            total_samples += pred.size(0)
        accuracy = correct_prediction.cpu().data.numpy()/total_samples
    return accuracy



#--------SAMPLER-------

class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_samples for each of the n_classes.
    Returns batches of size n_classes * (batch_size // n_classes)
    adapted from https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """

    def __init__(self, labels, batch_size):
        classes = sorted(set(labels.numpy()))
        print(classes)

        n_classes = len(classes)
        self._n_samples = batch_size // n_classes
        if self._n_samples == 0:
            raise ValueError(
                f"batch_size should be bigger than the number of classes, got {batch_size}"
            )

        self._class_iters = [
            InfiniteSliceIterator(np.where(labels == class_)[0], class_=class_)
            for class_ in classes
        ]

        batch_size = self._n_samples * n_classes
        self.n_dataset = len(labels)
        self._n_batches = self.n_dataset // batch_size
        if self._n_batches == 0:
            raise ValueError(
                f"Dataset is not big enough to generate batches with size {batch_size}"
            )
        print("K=", n_classes, "nk=", self._n_samples)
        print("Batch size = ", batch_size)

    def __iter__(self):
        for _ in range(self._n_batches):
            indices = []
            for class_iter in self._class_iters:
                indices.extend(class_iter.get(self._n_samples))
            np.random.shuffle(indices)
            yield indices

        for class_iter in self._class_iters:
            class_iter.reset()

    def __len__(self):
        return self._n_batches
    
    
class InfiniteSliceIterator:
    def __init__(self, array, class_):
        assert type(array) is np.ndarray
        self.array = array
        self.i = 0
        self.class_ = class_

    def reset(self):
        self.i = 0

    def get(self, n):
        len_ = len(self.array)
        # not enough element in 'array'
        if len_ < n:
            print(f"there are really few items in class {self.class_}")
            self.reset()
            np.random.shuffle(self.array)
            mul = n // len_
            rest = n - mul * len_
            return np.concatenate((np.tile(self.array, mul), self.array[:rest]))

        # not enough element in array's tail
        if len_ - self.i < n:
            self.reset()

        if self.i == 0:
            np.random.shuffle(self.array)
        i = self.i
        self.i += n
        return self.array[i : self.i]
