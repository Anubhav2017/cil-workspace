from model import Model
import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import argparse
import time
import numpy as np
import subprocess
from numpy import random
import copy
device="cpu"
from tqdm import tqdm

# import matplotlib.pyplot as plt
from data_loader import genomics_data

def kaiming_normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')


class Model(nn.Module):
    def __init__(self, classes):
        # Hyper Parameters
        self.init_lr = 0.1
        self.num_epochs = 40
        self.batch_size = 48
        
        self.pretrained = False
        self.momentum = 0.9
        self.weight_decay = 0.0001
        # Constant to provide numerical stability while normalizing
        self.epsilon = 1e-16

        # Network architecture
        super(Model, self).__init__()
        model = nn.Sequential(
        nn.Conv1d(1, 6, kernel_size=6),
        nn.ReLU(),
        nn.Conv1d(6, 3, kernel_size=6),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(12258,1024),
        nn.ReLU(),
        nn.Linear(1024,66),
        nn.Softmax()
        )

        self.model=model

        self.model.apply(kaiming_normal_init)
        

        self.n_classes = 0

    def forward(self, x):
        x = self.model(x)
        return x



    def classify(self, images):
        """Classify images by softmax

        Args:
            x: input image batch
        Returns:
            preds: Tensor of size (batch_size,)
        """
        _, preds = torch.max(torch.softmax(self.forward(images), dim=1), dim=1, keepdim=False)

        return preds

    def update(self, dataset, class_map,epoch):

        self.compute_means = True

        # Save a copy to compute distillation outputs


        classes = list(set(dataset.train_labels))

        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                shuffle=True, num_workers=8)

        print("Batch Size (for n_classes classes) : ", len(dataset))
        optimizer = optim.SGD(self.parameters(), lr=self.init_lr, momentum = self.momentum, weight_decay=self.weight_decay)
        for i, (indices, images, labels) in enumerate(loader):
            seen_labels = []
            images = Variable(torch.FloatTensor(images)).to(device)
            seen_labels = torch.LongTensor([class_map[label] for label in labels.numpy()])
            labels = Variable(seen_labels).to(device)
            # indices = indices.cuda()

            optimizer.zero_grad()
            logits = self.forward(images)

            loss = nn.CrossEntropyLoss()(logits, labels)

            loss.backward()
            optimizer.step()

            if (i+1) % 1 == 0:
                tqdm.write('Epoch [%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, i+1, np.ceil(len(dataset)/self.batch_size), loss.data))




model = Model(66)
model.to(device)
all_classes = np.arange(66)

train_set = genomics_data(train=True,classes=all_classes)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=48,shuffle=True, num_workers=8)

test_set = genomics_data(train=False,classes=all_classes)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=48,
                                            shuffle=False, num_workers=8)

for i in range(10):
    model.update(train_set, all_classes,i)

    total = 0.0
    correct = 0.0
    for indices, images, labels in train_loader:
        images = Variable(images).to(device)
        preds = model.classify(images)
        preds = [pred for pred in preds.cpu().numpy()]
        total += labels.size(0)
        correct += (preds == labels.numpy()).sum()

    # Train Accuracy
    print ('Train Accuracy : %.2f ,' % (100.0 * correct / total))

    total = 0.0
    correct = 0.0
    for indices, images, labels in test_loader:
        images = Variable(images).to(device)
        preds = model.classify(images)
        total += labels.size(0)
        correct += (preds == labels.numpy()).sum()

    # Test Accuracy
    print ('Test Accuracy : %.2f' % (100.0 * correct / total)) 