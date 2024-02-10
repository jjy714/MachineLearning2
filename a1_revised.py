import numpy as np
import torch
from torchvision import datasets
import torch.nn.functional as F
import time

trainset = datasets.MNIST(root='./data ', train=True, download=True)
testset = datasets.MNIST(root='./data ', train=False, download=True)

# Indices for train /val splits : train_idx , valid_idx
np.random.seed(0)
val_ratio = 0.1
train_size = len(trainset)
indices = list(range(train_size))
split_idx = int(np.floor(val_ratio * train_size))
np.random.shuffle(indices)
train_idx, val_idx = indices[split_idx:], indices[: split_idx]
train_data = trainset.data[train_idx].float()/255.
train_labels = trainset.targets[train_idx]
val_data = trainset.data[val_idx].float() / 255.
val_labels = trainset.targets[val_idx]
test_data = testset.data.float() / 255.
test_labels = testset.targets

train_size = len(train_idx)
val_size = len(val_idx)
test_size = len(testset)

ans_idx = 70

ans_label, ans_data = train_labels[ans_idx], train_data[ans_idx]


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
    distance_list = []
    distance_idx = []
    result_idx, temp = 0, 0
    for i in range(size):
        result = 0.0
        result = torch.sum((datas[i] - ansdata)**2)
        distance_list.append(result.item())
        distance_idx.append(labels[i])
    temp = distance_list.index(min(distance_list))
    result_idx = distance_idx[temp]

    return result_idx.item()


print("-------------------------")
print("Broadcasting start")
start_time = time.time()
broad_result = broad_classification(train_data, train_labels, train_size, ans_data)
print("Broadcasting classification classifies as: ", broad_result)
print("Answer label is: ", ans_label.item())
print("Time took for Broadcasting is: ", time.time() - start_time)
print("-------------------------")


# Sort original values and the matching indices
# get the list of the indices in  labels form
# find the most frequent label
# return value


def knn(datas, labels, size, anydigit, k):
    distances_tensor = torch.empty(1)
    distances_list = []
    distances_idx = []
    results = []
    for i in range(size):
        result = torch.sum((datas[i] - anydigit) ** 2)
        distances_list.append(result)
        distances_idx.append(labels[i])
    distances_tensor = torch.tensor(distances_list)
    temp, index = torch.sort(distances_tensor)
    index = index[:k]
    for j in index:
        results.append(distances_idx[j].item())
    end, _ = torch.mode(torch.tensor(results))

    return end.item()


print("-------------------------")
print("KNN start")
start_time = time.time()
knn_result = knn(train_data, train_labels, train_size, ans_data, k=5)
print("KNN classification classifies as: ", knn_result)
print("Answer label is: ", ans_label.item())
print("Time took for KNN is: ", time.time() - start_time)
print("-------------------------")


def knn_improved(datas, labels, size, anydigit, k):
    distances_list = []
    distances_idx = []
    results = []
    for i in range(size):
        distance = F.pairwise_distance(datas[i], anydigit)
        distances_list.append(torch.sum(distance))
        distances_idx.append(labels[i])
    distances_tensor = torch.tensor(distances_list)
    sorted_tensor, sorted_indices = torch.topk(distances_tensor, k, largest=False)
    for j in sorted_indices:
        results.append(distances_idx[j])
    result, _ = torch.mode(torch.tensor(results))

    return result.item()

print("-------------------------")
print("KNN improved start")
start_time = time.time()
knn_result = knn_improved(train_data, train_labels, train_size, ans_data, k=5)
print("KNN classification classifies as: ", knn_result)
print("Answer label is: ", ans_label.item())
print("Time took for KNN is: ", time.time() - start_time)
print("-------------------------")