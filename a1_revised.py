import numpy as np
import torch
from torchvision import datasets
import collections

trainset = datasets.MNIST(root='./ data ', train=True, download=True)
testset = datasets.MNIST(root='./ data ', train=False, download=True)


# Indices for train /val splits : train_idx , valid_idx
np. random . seed (0)
val_ratio = 0.1
train_size = len( trainset )
indices = list ( range ( train_size ))
split_idx = int(np. floor ( val_ratio * train_size ))
np. random . shuffle ( indices )
train_idx , val_idx = indices [ split_idx :], indices [: split_idx ]
train_data = trainset . data [ train_idx ]. float ()/255.
train_labels = trainset . targets [ train_idx ]
val_data = trainset . data [ val_idx ]. float ()/255.
val_labels = trainset . targets [ val_idx ]
test_data = testset . data . float ()/255.
test_labels = testset.targets

train_size = len(train_idx)
val_size = len(val_idx)
test_size = len(testset)

ans_label, ans_data = train_idx[6], train_data[6]

def loop_classification(datas, labels, size, ansdata):
    distance_list = []
    distance_idx = []
    ansdata = torch.flatten(ansdata)
    for i in range(size):
        subtraction, sum_sub = 0.0, 0.0
        data = torch.flatten(datas[i])
        for j in range(28*28):
            subtraction = data[j] - ansdata[j]
            sum_sub += torch.abs(subtraction)
        distance_list.append(sum_sub)
        distance_idx.append(labels[i])

    location = distance_list.index(min(distance_list))
    result = distance_idx[location]

    return result

def broad_classification(datas, labels, size, ansdata):
    distance_tensor = torch.zeros(0)
    distance_idx = []
    for i in range(size):
        result = datas[i] - ansdata
        result = torch.abs(result)
        torch.stack([distance_tensor, result])



    pass


