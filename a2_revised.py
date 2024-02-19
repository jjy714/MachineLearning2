import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torch.optim import SGD
import torch.nn
import time

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
    figure = plt.figure(figsize=(16, 16))
    cols, rows = 3, 3
    check_box = [0 for i in range(10)]
    img, label = 0, 0
    for i in range(1, cols * rows + 1):
        for j in range(len(trainset)):
            img, label = data[j]
            if check_box[label] == 1:
                break
            else:
                check_box[label] = 1
        # sample_idx = torch.randint(len(data), size=(1,)).item()
        # img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_dict[label])
        plt.axis("off")
        img = img.permute(1, 2, 0)
        plt.imshow(img.squeeze(), cmap='gray')
    plt.show()


visualize_image(trainset)

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
        loss = torch.mean(torch.max(zero, 1 - label * data))
        return loss



model = svm()
epoch = 10
batch_size = 64
learning_rate = 0.001
optimizer = SGD(model.parameters(), lr=learning_rate)



def train(model, optimizer):
    size = train_size
    n = 1
    model.train()

    for batch, (X, y) in enumerate(trainloader):
        label=0
        for i in y:
            if i == anyclass1:
                label = -1
            elif i == anyclass2:
                label = 1
            else:
                continue
        if label != anyclass1 and label != anyclass2: continue
        optimizer.zero_grad()
        prediction = model(X)
        loss = model.hingeloss(prediction, y)
        loss.backward()
        optimizer.step()
        if batch % batch_size == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Training loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def validation(model, optimizer):
    size = val_size
    n = 1
    for batch, (X, y) in enumerate(valloader):
        start_time = time.time()
        if y == anyclass1:
            y = -1
        elif y == anyclass2:
            y = 1
        else:
            continue
        optimizer.zero_grad()
        prediction = model.classifier(X)
        loss = model.hingeloss(prediction, y)
        loss.backward()
        optimizer.step()
        if batch % batch_size == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Training loss: {loss:>7f}  [{current:>5d}/{size:>5d}] at step{n}")

    n += 1



for epoch in range(epoch):
    n = 1
    start_time = time.time()
    print(f'Epoch {epoch + 1}')
    train(model, optimizer)
    print(f'Time took for step [{n}]: {(time.time() - start_time) / 60:>0.2f} mins')
    n += 1