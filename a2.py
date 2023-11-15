import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# (A) Load CIFAR10 dataset as follows:

trainset = datasets.CIFAR10(root='./ data ', train=True, download=True, transform=transforms.ToTensor())
testset = datasets.CIFAR10(root='./ data ', train=False, download=True, transform=transforms.ToTensor())

# (B) Visualize at least one image for each class. You may need to look into how dataset is
# implemented in PyTorch.
# shape of the image


# Get the labels and class names
# fig, ax= plt.subplots(nrows= 2, ncols= 5, figsize= (18,5))
# plt.suptitle('displaying one image of each category in train set'.upper(),y= 1.05, fontsize= 16)
#
# i= 0
# for j in range(2):
#     for k in range(5):
#         ax[j,k].imshow(trainset.data[list(trainset.targets).index(i)])
#         ax[j,k].axis('off')
#         ax[j,k].set_title(i)
#         i+=1
#
# plt.tight_layout()
# plt.show()


# (C) Split the trainset into training set and validation set with 90% : 10% ratio. Implement
# dataloaders for CIFAR10.


total_size = len(trainset)
train_size = int(0.9 * total_size)
val_size = total_size - train_size

# Use random_split to create training and validation datasets
train_dataset, val_dataset = random_split(trainset, [train_size, val_size])

train_data = torch.cat([train_dataset[i][0].unsqueeze(0) for i in range(len(train_dataset))], dim=0)
train_label = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])

val_data = torch.cat([val_dataset[i][0].unsqueeze(0) for i in range(len(val_dataset))], dim=0)
val_label = torch.tensor([val_dataset[i][1] for i in range(len(val_dataset))])
test_data = testset.data
test_label = testset.targets


# (D) Choose any two classes. Then, make a SVM classifier (implement a loss function yourself.
# Do not use PyTorch implementations of loss functions.) and its training/validation/evaluation
# code to perform binary classification between those two classes.

# class = 0, 2

class SVM:
    def __init__(self, train_data, train_label):
        self.w = torch.randn(3072, 1)
        self.b = torch.randn(1)
        self.y_prime = torch.where(train_label == 0, -1, 1)
        self.C = 1
    def hingeloss(self, x, y):
        # Regularizer term
        w = self.w
        b = self.b

        reg = 0.5 * (w * w)

        for i in range(x.shape[0]):
            # Optimization term
            x_ = x[i].view(3072, -1)
            opt_term = y[i] * ((w @ x_[i]) + b)

            # calculating loss
            loss = reg + self.C * max(0, 1 - opt_term)
        return loss[0][0]

SVM = SVM(train_data, train_label)
print(SVM.hingeloss(train_data, train_label))


# # (E) Train for 10 epochs with batch size 64.

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


def train_svm(X, y):
    svm = SVM()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=32)
    val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=32)

    for epoch in range(10):  # 훈련은 10 에포크 진행합니다.
        for X_batch, y_batch in train_loader:
            output = svm.forward(X_batch)
            loss = svm.hinge_loss(output, y_batch)
            loss.backward()
            svm.update_params()  # 업데이트

        # 검증 과정
        with torch.no_grad():
            val_loss = 0
            for X_batch, y_batch in val_loader:
                output = svm.forward(X_batch)
                val_loss += svm.hinge_loss(output, y_batch).mean().item()
            print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}")

    return svm  # 훈련된 모델을 반환합니다.


def evaluate_svm(svm, X_test, y_test):
    test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=32)
    correct = 0
    total = 0

    for X_batch, y_batch in test_loader:
        output = svm.forward(X_batch)
        predicted = torch.sign(output)  # 예측
        total += y_batch.size(0)  # 총 개수
        correct += (predicted == y_batch).sum().item()  # 맞춘 개수

    print('Accuracy of the network on the test data: %d %%' % (100 * correct / total))

# num_of_epochs = 10

# for epoch in range(num_of_epochs):
#     SVM.train(train_dataloader,test_dataloader)


# (F) Perform data normalization. You may need to look into how to use datasets in PyTorch.

# (G) Again, train for 10 epochs with batch size 64 after data normalization. Write down your
# observations.

# (H) What are the hyperparameters you can tune?

# (I) Try to obtain find optimal hyperparameters.

# (J) What is the final test accuracy?
