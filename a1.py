import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import time

trainset = datasets.MNIST(root='./data ', train=True, download=True)
testset = datasets.MNIST(root='./data ', train=False, download=True)

train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testset, batch_size=64, shuffle=True)

# Indices for train/val splits : train_idx , valid_idx
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
k=0



class knn():
    def __init__(self):
        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.k = k

    def loop_classification(self, num):
        new_train_example = torch.flatten(train_data[num])
        distance_list = []
        start_time = time.time()
        result_i = 0

        for i in range(len(val_data)):
            each_test = torch.flatten(val_data[i])
            distance = 0
            for j in range(len(new_train_example)):
                distance = distance + torch.sqrt(torch.sum((each_test[j] - new_train_example[j]) ** 2))
            distance_list.append(distance)

        result = min(distance_list)
        result_i = distance_list.index(result)

        print("the data is classified as: ", val_labels[result_i])
        print("the correct label is: ", train_labels[num])
        print("The time it takes to run the code is: ", time.time() - start_time)

        # Check if the classification is correct
        if train_labels[result_i] == val_labels[num]:
            print("Classification is correct!")
        else:
            print("Classification is incorrect.")

    def broadcasting_classification(self, num):
        d_list = []
        start_time = time.time()
        for i in val_data:
            distance = torch.sum((i - train_data[num]) ** 2)
            d_list.append(distance)
        lowest_distance = min(d_list)
        lowest_distance_idx = d_list.index(lowest_distance)

        print("the data is classified as: ", val_labels[lowest_distance_idx])
        print("the correct label is: ", train_labels[num])
        print("The time it takes to run the code is: ", time.time() - start_time)
        # print("All the distances are: ", sorted(d_list))
        if val_labels[lowest_distance_idx] == train_labels[num]:
            print("Classification is correct!")
        else:
            print("Classification is incorrect.")

    def classification(train_data, train_labels, val_data, val_labels, k):
        start_time = time.time()
        correct, incorrect, accuracy = 0, 0, 0

        for num, val_example in enumerate(val_data):
            d_list = []
            for i, train_example in enumerate(train_data):
                distance = torch.sum((val_example - train_example) ** 2)
                d_list.append((distance, train_labels[i]))

            # Sort the list by distance in ascending order
            d_list.sort(key=lambda x: x[0])

            # Get the labels of the k nearest neighbors
            k_nearest_labels = [label for _, label in d_list[:k]]

            # Find the most common label among the k nearest neighbors
            predicted_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
            print(int(num), "th iteration")
            print("the data is classified as: ", predicted_label)
            print("the correct label is: ", val_labels[num])
            if predicted_label == val_labels[num]:
                print("Classification is correct!")
                correct += 1
            else:
                print("Classification is incorrect.")
                incorrect += 1

        accuracy = (correct / (correct + incorrect)) * 100
        print("The accuracy is: ", accuracy, "%")
        print("The time it takes to run the code is: ", time.time() - start_time)
        return accuracy




    def train_validate(train_data, train_labels, val_data, val_labels, k):
        # Train the model (KNN in this case, so no actual training, just storing data)
        print("Training...")

        # Validation
        print("Validating...")
        accuracy_val = knn.classification(train_data, train_labels, val_data, val_labels, k)
        print(accuracy_val)


    def train_evaluate(train_data, train_labels, test_data, test_labels, k):
        # Train the model (KNN in this case, so no actual training, just storing data)
        print("Training...")

        # Evaluate on the test set
        print("Evaluating...")
        accuracy_test = knn.classification(train_data, train_labels, test_data, test_labels, k)
        print(accuracy_test)


    def improved_classification(train_data, train_labels, val_data, val_labels, k):
        start_time = time.time()
        correct, incorrect, accuracy = 0, 0, 0

        for num, val_example in enumerate(val_data):
            distances = torch.norm(val_example - train_data, dim=1, p=2)
            print(distances)
            _, indices = torch.topk(distances, k, largest=False)
            print(indices)
            k_nearest_labels = train_labels[indices]
            print(k_nearest_labels)
            predicted_label = torch.mode(k_nearest_labels)

            if predicted_label == val_labels[num].item():
                print("Classification is correct!")
                correct += 1
            else:
                print("Classification is incorrect.")
                incorrect += 1

        accuracy = (correct / (correct + incorrect)) * 100
        print("The accuracy is: {:.2f}%".format(accuracy))
        print("The time it takes to run the code is: {:.4f} seconds".format(time.time() - start_time))
        return accuracy

# Example usage
# train_validate(train_data, train_labels, val_data, val_labels, 7)
# train_evaluate(train_data, train_labels, test_data, test_labels, 7)
# # loop_classification(0)
# broadcasting_classification(0)
# classification(train_data, train_labels, val_data, val_labels, k=15)
# train_evaluate(train_data, train_labels, test_data, test_labels, k=5)
#knn.improved_classification(train_data, train_labels, val_data, val_labels, k=5)
# knn.improved_classification(train_data, train_labels, val_data, val_labels, k=3)
