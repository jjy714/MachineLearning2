
import numpy as np
import torch
from torchvision import transforms , datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import nn

#(A) Load CIFAR10 dataset as follows:






trainset = datasets.CIFAR10(root='./ data ', train=True, download=True, transform=transforms.ToTensor())
testset = datasets.CIFAR10(root='./ data ', train=False, download=True, transform=transforms.ToTensor())

# (C) Split the trainset into training set and validation set with 90% : 10% ratio. Implement
# dataloaders for CIFAR10.

np.random.seed(0)
val_ratio = 0.1
train_size = len(trainset)
indices = list(range(train_size))
split_idx = int(np.floor(val_ratio * train_size))
np.random.shuffle(indices)
train_idx, val_idx = indices[split_idx:], indices[:split_idx]

train_data = trainset.data[train_idx].float() / 255.
train_labels = trainset.targets[train_idx]
val_data = trainset.data[val_idx].float() / 255.
val_labels = trainset.targets[val_idx]
test_data = testset.data.float() / 255.
test_labels = testset.targets



# (D) Choose any two classes. Then, make a SVM classifier (implement a loss function yourself.
# Do not use PyTorch implementations of loss functions.) and its training/validation/evaluation
# code to perform binary classification between those two classes.
class SVM():
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def train(self, X, Y):
        w = torch.randn(3072, 1, requires_grad=True)
        b = torch.randn(1, requires_grad=True)
        gamma = 10
        loss = [Y[i]*(X[i]*w+b)>=1 for i in range(len(X))]
        obj = torch.min(0.5 * torch.norm(w) ** 2 + gamma * torch.sum(loss), torch.tensor(0.0))

        return obj


# (E) Train for 10 epochs with batch size 64.
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

num_of_epochs = 10

for epoch in range(num_of_epochs):
    SVM.train(train_dataloader,test_dataloader)


# (F) Perform data normalization. You may need to look into how to use datasets in PyTorch.

# (G) Again, train for 10 epochs with batch size 64 after data normalization. Write down your
# observations.

# (H) What are the hyperparameters you can tune?

# (I) Try to obtain find optimal hyperparameters.

# (J) What is the final test accuracy?

