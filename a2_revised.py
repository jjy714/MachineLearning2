import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torch.optim import SGD
import torch.nn
import time
import numpy as np
import random

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
    check_box = [0 for i in range(10)]
    img, label = 0, 0
    for i in range(1, cols * rows + 1):
        for j in range(len(trainset)):
            img, label = data[j]
            if check_box[label] == 1:
                continue
            else:
                check_box[label] = 1
                break
        figure.add_subplot(rows, cols, i)
        plt.title(labels_dict[label])
        plt.axis("off")
        img = img.permute(1, 2, 0)
        plt.imshow(img.squeeze(), cmap='gray')
    plt.show()


# visualize_image(trainset)

total_size = len(trainset)
train_size = int(0.9 * total_size)
val_size = int(0.1 * total_size)

train_dataset, val_dataset = random_split(trainset, [train_size, val_size])

# data shape[0] = (1, 3, 32, 32)
# label shape[0] = (1)

trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valloader = DataLoader(val_dataset, shuffle=True)
testloader = DataLoader(testset, shuffle=False)

anyclass1, anyclass2 = 3, 7


class svm(torch.nn.Module):
    def __init__(self):
        super(svm, self).__init__()
        self.fc = torch.nn.Linear(32, 1)

    def forward(self, x):
        fd = self.fc(x)
        return fd

    def hingeloss(self, data, label):
        zero = torch.zeros(1)
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
        loss = torch.mean(torch.clamp(1 - data * label, min=0))
        return loss


model = svm()
epoch = 10
batch_size = 64
learning_rate = 0.001
optimizer = SGD(model.parameters(), lr=learning_rate)


def train(model, optimizer, datas):
    size = train_size
    n = 1
    model.train()

    for batch, (X, y) in enumerate(datas):
        label = 0
        for i in y:
            if i == anyclass1:
                label = 1
            elif i == anyclass2:
                label = -1
            else:
                continue

        optimizer.zero_grad()
        prediction = model(X)
        loss = model.hingeloss(prediction, label)
        loss.backward()
        optimizer.step()
        if batch % batch_size == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Training loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validation(model):
    size = val_size
    n = 1
    model.eval()
    val_loss, correct = 0, 0
    num_batches = len(valloader)

    with torch.no_grad():
        for X, y in valloader:
            if y == anyclass1:
                y = 1
            elif y == anyclass2:
                y = -1
            else:
                continue
            pred = model(X)
            val_loss += model.hingeloss(pred, y.float()).item()
            correct += ((pred > 0.5) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")


# def classification(model, data):
#     list_a = []
#
#     label = 0
#     datas, label2 = data
#     if label2 == anyclass1:
#         label = 1
#     elif label2 == anyclass2:
#         label = -1
#     else:
#         return False
#     print(label * model(datas))
#     _, result = label * model(datas)
#     ans = torch.max (result, dim=1)
#     return ans

# for i in range(10):
#     idx = random.randint(0, 2000)
#     print(f"{i + 1}th attempt")
#     print(f"{idx}'s label")
#     print(classification(model, trainloader.dataset[idx]))


# for epoch in range(epoch):
#
#     print("training without normalization")
#     start_time = time.time()
#     print(f'Epoch {epoch + 1}')
#     train(model, optimizer, trainloader)
#     print(f'Time took for step [{epoch}]: {(time.time() - start_time) / 60:>0.2f} mins')

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

#
trainset = datasets.CIFAR10(root='./CIFARdata', train=True, download=True, transform=train_transform)
train_dataset, val_dataset = random_split(trainset, [train_size, val_size])

trainloader2 = DataLoader(train_dataset, batch_size=64, shuffle=True)
valloader = DataLoader(val_dataset, shuffle=True)





print("Normalized data training")
for epoch in range(epoch):
    start_time = time.time()
    print(f'Epoch {epoch + 1}')
    train(model, optimizer, trainloader2)
    print(f'Time took for step [{epoch}]: {(time.time() - start_time) / 60:>0.2f} mins')

validation(model)