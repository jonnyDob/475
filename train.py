import torch
import random

# import necessary libraries
import torch.nn as nn
import sys
import argparse
import torchsummary
from custom_dataset import custom_dataset
import tqdm
import torch
from torch.utils.data import DataLoader, Subset
#from custom_dataset import custom_dataset
import torch.optim as optim
import datetime as dt
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from network import Resnet18
print(torch.cuda.is_available())

gamma_idx = sys.argv.index("-gamma")
epoch_idx = sys.argv.index("-e")
batch_idx = sys.argv.index("-b")
loaddataset_idx = sys.argv.index("-l")
classifier_idx = sys.argv.index("-s")
savefile_idx = sys.argv.index("-p")
train_idx = sys.argv.index("-train")
cuda_idx = sys.argv.index("-cuda")
gamma = float(sys.argv[gamma_idx + 1])
epoch = int(sys.argv[epoch_idx + 1])
batch_size = int(sys.argv[batch_idx + 1])
load_dataset = sys.argv[loaddataset_idx + 1]
classifierFile = sys.argv[classifier_idx + 1]
savefile = sys.argv[savefile_idx + 1]
cuda = sys.argv[cuda_idx + 1]
device = torch.device(cuda)
trainingMode = sys.argv[train_idx + 1]


def train_transform():
    transform_list = [
        # transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

model = Resnet18()
model.train()
model.to(device=device)

train_tf=train_transform()
train_dataset=custom_dataset(load_dataset, train_tf)

optimizer=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

num_batches = int(len(train_dataset) / batch_size)

train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

def train(model, train_loader, optimizer, scheduler, epochs, device=device):
    print("Training...")
    model.train()
    losses_train = []
    model.to(device=device)
    i = 0
    for epoch in tqdm.tqdm(range(epochs)):
        loss_train = 0.0

        for imgs, labels in train_loader:

            imgs = imgs.to(device=device)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            labels_tensor = labels_tensor.to(device=device)
            # print("\n\n This is the shape of the imgs: \n")
            # print(imgs.shape)

            i = i + 1
            print(i)
            output = model(imgs)
            loss = model.lossFunction(output, labels_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        scheduler.step()
        losses_train += [loss_train / (num_batches)]

        print('Epoch {}, Training loss {}'.format(epoch, loss_train / (num_batches)))
        state_dict = model.classifier.state_dict()
        torch.save(state_dict, classifierFile)

    plt.plot(losses_train, label='Total Loss')
    plt.legend()
    plt.savefig(savefile)
    plt.show()

    return losses_train

if trainingMode == 'y':
    train(model=model, train_loader = train_loader, optimizer=optimizer, scheduler=scheduler, epochs=epoch)
