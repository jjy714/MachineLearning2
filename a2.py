import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# (A) Load CIFAR10 dataset as follows:

trainset = datasets.CIFAR10(root='./ data ', train=True, download=True, transform=transforms.ToTensor())
testset = datasets.CIFAR10(root='./ data ', train=False, download=True, transform=transforms.ToTensor())


# (B) Visualize at least one image for each class. You may need to look into how dataset is
# implemented in PyTorch.
# shape of the image


iters = iter(trainset)
imgs, labels = iters.next()
for i in range(10):
    if label == i:
        plt.imshow(transforms.ToPILImage()(iters.next()[0]))
    label = labels[i]
    # for img, label in trainset


plt.show()

# label names
# take first image
# image = data_batch_1['data'][0]
# # take first image label index
# label = data_batch_1['labels'][0]
# # Reshape the image
# image = image.reshape(3,32,32)
# # Transpose the image
# image = image.transpose(1,2,0)
# # Display the image
# plt.imshow(image)
# plt.title(label_name[label])
#

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

# class = 0, 2


class SVM:
    def __init__(self):
        self.w = torch.randn(3072, 1, requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)
        self.gamma = 10
        self.ex_class1 = 0
        self.ex_class2 = 2

    def train(self, X, Y):
        for i in X:
            hinge_loss = torch.max(0, 1 - Y[self.ex_class2] * (i @ self.w + self.b))


        regularization_term = 0.5 * torch.norm(self.w) ** 2
        obj = regularization_term + self.gamma * torch.sum(hinge_loss)

        return obj

    def predict(self, X):
        pass

    def accuracy(self, X, Y):
        pass



#
# # Instantiate the SVM model
# svm_model = SVM()
#
# # Example usage
# X_train = torch.randn((100, 3072))  # Replace with your actual training data
# Y_train = torch.randint(0, 2, (100, 1)) * 2 - 1  # Binary labels (-1 or 1)
#
# # Train the SVM
# loss = svm_model.train(X_train, Y_train)
#
# # Perform optimization steps here (missing in the provided code)
# # Use torch.optim and optimizer.step() to update weights based on gradients
#
#
# # (E) Train for 10 epochs with batch size 64.
# train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
#
# num_of_epochs = 10

# for epoch in range(num_of_epochs):
#     SVM.train(train_dataloader,test_dataloader)


# (F) Perform data normalization. You may need to look into how to use datasets in PyTorch.

# (G) Again, train for 10 epochs with batch size 64 after data normalization. Write down your
# observations.

# (H) What are the hyperparameters you can tune?

# (I) Try to obtain find optimal hyperparameters.

# (J) What is the final test accuracy?
