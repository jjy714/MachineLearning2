import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

trainset = datasets.CIFAR10(root='./CIFARdata', train=True, download=True, transform=transforms.ToTensor())
testset = datasets.CIFAR10(root='./CIFARdata', train=False, download=True, transform=transforms.ToTensor())

total_size = len(trainset)
train_size = int(0.9 * total_size)
val_size = int(0.1 * total_size)

train_dataset, val_dataset = random_split(trainset, [train_size, val_size])

# data shape[0] = (1, 3, 32, 32)
# label shape[0] = (1)
trainloader = DataLoader(train_dataset, shuffle=True)
valloader = DataLoader(val_dataset, shuffle=True)
testloader = DataLoader(testset, shuffle=False)


def visualize_image(data):
    global img, label
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    labels_dict = {0: 'airplane',
                   1: 'automobile',
                   2: 'bird',
                   3: 'cat',
                   4: 'deer',
                   5: 'dog',
                   6: 'frog',
                   7: 'horse',
                   8: 'ship',
                   9: 'truck'}

    for i in range(1, cols * rows + 1):
        overlapped = False
        label_overlap = 0
        while not overlapped:
            sample_idx = torch.randint(len(data), size=(1,)).item()
            img, label = data[sample_idx]
            if label != label_overlap:
                overlapped = False
            label_overlap = label
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

visualize_image(trainloader)
