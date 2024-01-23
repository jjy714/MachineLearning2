import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


trainset = datasets.CIFAR10(root='./ data ', train=True, download=True, transform=transforms.ToTensor())
testset = datasets.CIFAR10(root='./ data ', train=False, download=True, transform=transforms.ToTensor())


def visualize_image(trainset):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(trainset), size=(1,)).item()
        img, label = trainset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

visualize_image(trainset)

train_loader = DataLoader(trainset)
test_loader = DataLoader(testset)









