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
import torch.nn.functional as F
import torch.utils.data
import random
import numpy as np

from torch.utils.data.sampler import BatchSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#-------- Eval function --------

def model_eval(dataloader, model_g, model_f):
    """
    Model evaluation function
    args:
    - dataloader : considered dataset
    - model_g : feature exctrator (torch.nn)
    - model_f : classification layer (torch.nn)
    """
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
    Taken from https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/sampler.py
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
