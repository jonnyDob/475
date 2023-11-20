import torch
import random

# import necessary libraries
import torch.nn as nn
import sys
import argparse
import torchsummary
import torchvision.models

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
        transforms.Resize(size=(150, 150)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

# Creation of the model
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)

lossFunction = nn.BCEWithLogitsLoss()



# model.train()
# model.to(device=device)





train_tf=train_transform()
test_tf=train_transform()

train_dataset = custom_dataset(load_dataset, train_tf)
test_dataset = custom_dataset(load_dataset, test_tf)

train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
testLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

num_batches = int(len(train_dataset) / batch_size)

def train(model, train_loader, optimizer, scheduler, epochs, device=device):
    print("Training...")
    model.train()
    losses_train = []
    model.to(device=device)
    i = 0
    for epoch in tqdm.tqdm(range(epochs)):
        loss_train = 0.0

        for imgs, labels in train_loader:

            # TODO change the type for it to work back to , dtype=torch.int64 if needed
            imgs = imgs.to(device=device)
            labels = torch.tensor(labels, dtype=torch.float32).to(device=device)


            # labels = labels.to(device=device)
            # labels_tensor = torch.tensor(labels)
            # labels_tensor = labels_tensor.to(device=device)
            # print("\n\n This is the shape of the imgs: \n")
            # print(imgs.shape)

            # imgs.to(device=device)
            # labels_tensor = torch.tensor(labels)
            # labels_tensor.to(device=device)


            i = i + 1
            print(i)
            output = model(imgs)
            # print("\n\n This is the output: ")
            # print(output)
            # print("\n THIS IS THE END OF OUTPUT")
            output = output.squeeze()
            # print("\n\n This is the output AFTER: ")
            # print(output)
            # print("\n THIS IS THE END OF OUTPUT AFTER")
            # labels.float()
            loss = lossFunction(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        scheduler.step()
        losses_train += [loss_train / (num_batches)]

        print('Epoch {}, Training loss {}'.format(epoch, loss_train / (num_batches)))
        state_dict = model.state_dict()
        torch.save(state_dict, classifierFile)

    plt.plot(losses_train, label='Total Loss')
    plt.legend()
    plt.savefig(savefile)
    plt.show()

    return losses_train


def test(model, loader, device):
    print("Testing...")
    model.eval()
    model.to(device=device)
    accuracy = Accuracy(top_k=1, num_classes= 2, task='binary').to(device=device)

    with torch.no_grad():
        i=0
        for imgs, labels in loader:

            imgs = imgs.to(device=device)
            labels_tensor = torch.tensor(labels)
            labels_tensor = labels_tensor.to(device=device)

            output = model.forward(imgs)
            output=output.squeeze()
            # output = model(imgs)

            print(i)
            i=i+1


            accuracy.update(output,labels_tensor)

    theAccuracy = accuracy.compute()

    print(f"Top-1 Accuracy: {theAccuracy.item() * 100:.2f}%")

if trainingMode == 'y':
    train(model=model, train_loader = train_loader, optimizer=optimizer, scheduler=scheduler, epochs=epoch)
else:
    classifier_state_dict = torch.load(classifierFile)
    model.load_state_dict(classifier_state_dict)
    test(model = model, loader = testLoader, device = device)