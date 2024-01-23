import numpy as np
import torch
from torchvision import datasets
import time

trainset = datasets.MNIST(root='./ data ', train=True, download=True)
testset = datasets.MNIST(root='./ data ', train=False, download=True)

# Indices for train /val splits : train_idx , valid_idx
np.random.seed(0)
val_ratio = 0.1
train_size = len(trainset)
indices = list(range(train_size))
split_idx = int(np.floor(val_ratio * train_size))
np.random.shuffle(indices)
train_idx, val_idx = indices[split_idx:], indices[: split_idx]
train_data = trainset.data[train_idx].float() / 255.
train_labels = trainset.targets[train_idx]
val_data = trainset.data[val_idx].float() / 255.
val_labels = trainset.targets[val_idx]
test_data = testset.data.float() / 255.
test_labels = testset.targets

train_size = len(train_idx)
val_size = len(val_idx)
test_size = len(testset)

ans_label, ans_data = train_labels[6], train_data[6]


def loop_classification(datas, labels, size, ansdata):
    distance_list = []
    distance_idx = []
    ansdata = torch.flatten(ansdata)
    for i in range(size):
        subtraction, sum_sub = 0.0, 0.0
        data = torch.flatten(datas[i])
        for j in range(28 * 28):
            subtraction = data[j] - ansdata[j]
            sum_sub += torch.abs(subtraction)
        distance_list.append(sum_sub)
        distance_idx.append(labels[i])

    location = distance_list.index(min(distance_list))
    result = distance_idx[location]

    return result


# print("-------------------------")
# print("Loop start")
# start_time = time.time()
# loop_result = loop_classification(train_data, train_labels, train_size, ans_data)
# print("Answer data's label: ", ans_label)
# print("Loop classification classifies as: ", loop_result)
# print("Time took for Loop is: ", time.time() - start_time)
# print("-------------------------")
#

def broad_classification(datas, labels, size, ansdata):
    distance_tensor = torch.empty(0)
    distance_idx = []
    result_idx, temp = 0, 0
    for i in range(size):
        result = 0.0
        result = torch.sum((datas[i] - ansdata)**2)
        torch.cat([distance_tensor, result.reshape(1)], 0)
        distance_idx.append(labels[i])
    temp = torch.argmin(distance_tensor)
    temp = temp.item()
    result_idx = distance_idx[temp]

    return result_idx


print("-------------------------")
print("Broadcasting start")
start_time = time.time()
broad_result = broad_classification(train_data, train_labels, train_size, ans_data)
print("Broadcasting classification classifies as: ", broad_result)
print("Time took for Broadcasting is: ", time.time() - start_time)
print("-------------------------")


# Sort original values and the matching indices
# get the list of the indices in  labels form
# find the most frequent label
# return value


def knn(datas, labels, size, anydigit, k):
    distances_tensor = torch.empty(0)
    distances_idx = []
    results = []
    for i in range(size):
        result = torch.sum((datas[i] - anydigit) ** 2)
        torch.cat([distances_tensor, result.reshape(1)])
        distances_idx.append(labels[i])
    temp, index = torch.sort(distances_tensor)
    index = index[:k]
    for j in index:
        results = distances_idx[j]
    end = torch.mode(results)

    return end


print("-------------------------")
print("KNN start")
start_time = time.time()
knn_result = knn(train_data, train_labels, train_size, ans_data, k=5)
print("KNN classification classifies as: ", knn_result)
print("Time took for KNN is: ", time.time() - start_time)
print("-------------------------")


def knn_improved(datas, labels, size, anydigit, k):
    distances_tensor = torch.empty(0)
    distances_idx = []
    results = []
    for i in range(size):
        distance = torch.cdist(datas[i], anydigit)
        torch.cat([distances_tensor, distance.reshape(1)])
        distances_idx.append(labels[i])
    sorted_tensor, sorted_indices = torch.topk(distances_tensor, k)
    for j in sorted_indices:
        results = distances_idx[j]
    result = torch.mode(results)

    return result
