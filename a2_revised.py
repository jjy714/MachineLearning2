import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torch.optim import SGD

trainset = datasets.CIFAR10(root='./CIFARdata', train=True, download=True, transform=transforms.ToTensor())
testset = datasets.CIFAR10(root='./CIFARdata', train=False, download=True, transform=transforms.ToTensor())

labels_dict = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'}


def visualize_image(data):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_dict[label])
        plt.axis("off")
        plt.imshow(img.T.squeeze(), cmap="gray")
    plt.show()


visualize_image(trainset)

total_size = len(trainset)
train_size = int(0.9 * total_size)
val_size = int(0.1 * total_size)

train_dataset, val_dataset = random_split(trainset, [train_size, val_size])

# data shape[0] = (1, 3, 32, 32)
# label shape[0] = (1)
trainloader = DataLoader(train_dataset, shuffle=True)
valloader = DataLoader(val_dataset, shuffle=True)
testloader = DataLoader(testset, shuffle=False)

anyclass1, anyclass2 = 3, 7
ans_data1, ans_label1 = enumerate(trainloader[anyclass1])
ans_data2, ans_label2 = enumerate(trainloader[anyclass2])


# torch.where

# if yi >=1 -> target1
# if yi <= -1 -> target2

# if ans (1 or -1) and data label calcuation results in a correct answer, pass
# else, if the result is not matching, adjust the w and b value.


# update w and b to calculate to a result of 1
class svm(torch.nn.Module):
    def __init__(self):
        super(svm, self).__init__()


    def classifier(self, data):
        w, b = self.parameters()
        result = data * w + b
        return torch.sign(result)
    def data_size(self, data):
        if data == 'train':
            return train_size
        elif data == 'validation':
            return val_size
        elif data == 'test':
            return len(testset)

    def hingeloss(self, data, label):
        w, b = self.parameters()
        zero = self.zero
        data_size = len(data)

        regularizer = 0.5 * w * w
        # for i in range(data_size):
        #     ans = -1
        #     if data[i] is not ans_label1 or ans_label2:
        #         continue
        #     else:
        #         if data[i] is ans_label1:
        #             ans = 1
        #             result = torch.max(zero, 1 - ans * (w.T * data - b))
        #         if data[i] is ans_label2:
        #             result = torch.max(zero, 1 - ans * (w.T * data - b))
        for i in range(data_size):
            term = label[i] * (w.T * data[i]) + b
            loss = regularizer + self.c * torch.max(zero, 1 - term)

        return loss

    def loss(self):
        # t = yfx
        # if t >= 1 -> answer correct, no need to weigh error
        # if 0 < t < 1 -> answer not correct but within error margin, grant 1 - t error
        # if t < 0 -> answer is wrong. Grant 1 - t error.

        # IN function, => h(t) = max(0, 1-t)

        pass

    def parameters(self):
        w = torch.rand()
        b = torch.rand()
        return self.w, self.b



model = svm()
epoch = 10
batch_size = 64
learning_rate = 0.1

optimizer = SGD(model.parameters(),lr=learning_rate )
loss_fn = model.hingeloss()
optimizer.step()



def train():
    size = train_size

    for epochs in range(epoch):
        for X, y in trainloader:
            optimizer.zero_grad()
            prediction = svm.classifier(X)
            loss = model.hingeloss(X, y)
            loss.backward()
            optimizer.step()
        return loss

def validation():
    size = val_size

    for epochs in range(epoch):
        for X, y in valloader:
            optimizer.zero_grad()
            prediction = svm.classifier(X)
            loss = model.hingeloss(X, y)
            loss.backward()
            optimizer.step()
        return loss
