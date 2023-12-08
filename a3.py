import torch
from torchvision import transforms
from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
import glob


class CustomImageDataset(Dataset):

    def read_data(self):
        train_label = pd.read_csv('/Users/jason/Projects/MachineLearningProjects/klanguage/train.csv')
        _1 = []
        _2 = []
        train_idx = []
        val_idx = []
        train_size = len(train_label)
        self.val_ratio = 0.2
        split_idx = int(np.floor(self.val_ratio * train_size))
        for i in train_label.values:
            _1.append(i[0])
            train_idx.append(i[1])

        train_idx, val_idx = train_idx[split_idx:], train_idx[:split_idx]

        return train_idx, val_idx

    def getLen(self, data_path, train):
        n = 0
        if train:
            for i in glob.glob(data_path + '/*'):
                n += 1
            return int(np.floor(n * (1 - self.val_ratio)))
        else:
            for i in glob.glob(data_path + '/*'):
                n += 1
            return int(np.floor(n * self.val_ratio))

    def __init__(self, root_directory, train, transform=None):
        super(CustomImageDataset, self).__init__()
        self.root_directory = root_directory
        self.transform = transform
        self.train = train
        self.train_label, self.val_label = self.read_data()
        self.val_ratio = 0.2

    def __len__(self):
        if self.train:
            return self.getLen(self.root_directory, train=True)
        else:
            return self.getLen(self.root_directory, train=False)

    def __getitem__(self, idx):
        if self.train:
            desired_width = 3
            number_str = str(idx).zfill(desired_width)
            img = Image.open(train_data_path + f'/{number_str}.jpg')
            if transforms is not None:
                img = self.transform(img)
            return img, self.train_label[idx]
        else:
            desired_width = 3
            number_str = str(idx).zfill(desired_width)
            img = Image.open(train_data_path + f'/{number_str}.jpg')
            if transforms is not None:
                img = self.transform(img)
            return img, self.val_label[idx]


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L4 FC 4x4x128 inputs -> 625 outputs
        self.layer4 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        self.relu1 = torch.nn.ReLU()
        # L5 Final FC 625 inputs -> 10 outputs
        self.layer5 = torch.nn.Linear(625, 11, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # Flatten them for FC
        out = self.layer4(out)
        out = self.layer5(out)
        return out


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

learning_rate = 0.001
training_epochs = 10
batch_size = 64

train_data_path = '/Users/jason/Projects/MachineLearningProjects/klanguage/train'
test_data_path = '/Users/jason/Projects/MachineLearningProjects/klanguage/test'
train_data = CustomImageDataset(train_data_path, train=True, transform=train_transform)
val_data = CustomImageDataset(train_data_path, train=False, transform=val_transform)
train_loader = DataLoader(train_data, batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size, shuffle=False)


def train_loop(modelc, loss_fun, optimizerc):
    size = train_data.__len__()
    modelc.train()
    for batch, (X, y) in enumerate(train_loader):
        # Compute prediction and loss
        pred = modelc(X)
        loss = loss_fun(pred, y)
        # Backpropagation
        optimizerc.zero_grad()
        loss.backward()
        optimizerc.step()

        if batch % 64 == 0:
            print(f'Training loss (for one batch) at step {batch + 1}: {float(loss)}')
            print(f'Seen so far: {((batch + 1) * size)} samples')
            batch += 1


def test_loop(modelc, loss_fun):
    modelc.eval()
    size = val_data.__len__()
    num_batches = len(val_loader)
    validity_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in val_loader:
            pred = modelc(X)

            validity_loss += loss_fun(pred, y).item()
            _, predicted = torch.max(pred.data, 1)
            correct += ((predicted > 0.5) == y).sum().item()

        validity_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {validity_loss:>8f} \n")


model = CNN()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()
# for t in range(training_epochs):
#     print(f"Epoch {t + 1}\n-------------------------------")
#     train_loop(model, loss_fn, optimizer)
# print("Done!")

test_loop(model, loss_fn)
