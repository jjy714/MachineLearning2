
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

trainset = datasets.MNIST(root='./data ', train=True, download=True)
testset = datasets.MNIST(root='./data ', train=False, download=True)

train_dataloader = DataLoader(trainset, batch_size=64, shuffle = True)
test_dataloader = DataLoader(testset, batch_size=64, shuffle = True)



# Indices for train/val splits : train_idx , valid_idx
np.random.seed(0)
val_ratio = 0.1
train_size = len( trainset )
indices = list ( range ( train_size ))
split_idx = int(np. floor ( val_ratio * train_size ))
np.random.shuffle ( indices )
train_idx, val_idx = indices[split_idx:], indices[:split_idx]

train_data = trainset.data[train_idx].float()/255.
train_labels = trainset.targets[train_idx]
val_data = trainset.data[val_idx].float()/255.
val_labels = trainset.targets[val_idx]
test_data = testset.data.float()/255.
test_labels = testset.targets


def loop_classification(num):
    new_train_example = torch.flatten(train_data[num])
    distance_list = []
    result_i = 0

    for i in range(len(test_data)):
        each_test = torch.flatten(test_data[i])
        # Calculate the Euclidean distance
        distance = torch.sqrt(torch.sum((each_test - new_train_example)**2))
        distance_list.append(distance)

    # Find the index of the minimum distance
    result_i = torch.argmin(torch.tensor(distance_list))

    # Print the results
    print("the data is classified as: ", test_labels[result_i])
    print("the correct label is: ", val_labels[num])

    # Check if the classification is correct
    if test_labels[result_i] == val_labels[num]:
        print("Classification is correct!")
    else:
        print("Classification is incorrect.")

    # Print additional labels
    print(val_labels[num])
    print(train_labels[num])

# Assume you have defined train_data, test_data, train_labels, test_labels, val_labels
loop_classification(0)


def broadcasting_classification(num):
    d_list = []
    start_time = time.time()
    for i in test_data:
        distance = torch.sum((i - train_data[num])**2)
        d_list.append(distance)
    lowest_distance = min(d_list)
    lowest_distance_idx = d_list.index(lowest_distance)
    print("the data is classified as: ", test_labels[lowest_distance_idx])
    print("the correct label is: ", val_labels[num])
    print("The time it takes to run the code is: ", time.time() - start_time)


broadcasting_classification(0)