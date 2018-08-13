import torch, torchvision
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import glob
import os
import math
import random
import argparse

BATCH_SIZE = 64
variables = ['dens']
NX, NY = 480, 480
CONV_INPUT_SHAPE = (len(variables), NX, NY)


class FlashDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_files= []
        self.targets = []

        # Add error data
        error_data_files = list(glob.iglob(data_dir+"/error/*"))
        self.targets.append(np.ones((len(error_data_files), 1)))

        # Add clean data
        clean_data_files = list(glob.iglob(data_dir+"/clean/*"))
        self.targets.append(np.zeros((len(clean_data_files), 1)))

        self.data_files = error_data_files+clean_data_files
        self.targets = np.vstack(self.targets)
        print self.targets.shape
        self.targets = torch.from_numpy(self.targets)
        print "data size:", len(self.data_files), ", targets size: ", self.targets.size()

    def __getitem__(self, index):
        return self.read_binary(self.data_files[index]), self.targets[index]
    def __len__(self):
        return self.targets.size(0)
    def read_binary(self, filename):
        data = np.fromfile(filename, dtype=np.double).reshape(1, NX, NY)
        return torch.from_numpy(data)

class FlashNet(nn.Module):
    def __init__(self):
        super(FlashNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(len(variables), 32, 5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1),
        )
        conv_output_size = self.get_conv_output_size()
        print "conv output size: ", conv_output_size
        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_output_size, out_features=1, bias=True),
            #nn.ReLU(),
            #nn.Dropout(p=0.2),
            #nn.Linear(768, 512),
            #nn.ReLU(),
            #nn.Dropout(p=0.5),
            #nn.Linear(512, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(x.size(0), -1)   # flatten
        return self.fc.forward(x)

    # Helper function: find out the output size of conv layer
    # So we can pass it to the linear layer
    def get_conv_output_size(self):
        conv_input = Variable(torch.rand(BATCH_SIZE, *CONV_INPUT_SHAPE))
        conv_output = self.conv.forward(conv_input)
        output_size = conv_output.data.view(BATCH_SIZE, -1).size(1)
        return output_size

def training(model, train_loader, epochs=5, use_gpu=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #optimizer = torch.optim.SGD(model.parameters(), lr = 1e-6, momentum=0.5)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
    #loss_func = nn.BCELoss()

    running_loss = 0
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            inputs, labels = data

            samples = labels.size()[0]
            weights = torch.DoubleTensor([2.0]*samples).view(-1, 1)  # shape of (BATCH_SIZE, 1)

            if torch.cuda.is_available() and use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
                weights = weights.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = model(inputs)

            # This makes class 1 has less weights
            weights = weights - 1.0 * labels.data
            loss_func = nn.BCELoss(weights)

            loss = loss_func(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.data[0]
            if i % 20 == 0 and i != 0:
                print("epoch:%s i:%s loss:%s" %(epoch, i, running_loss/20))
                running_loss = 0

def evaluating(model, test_loader, use_gpu=True):
    num_correct = 0.0
    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        if torch.cuda.is_available() and use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        output = model(inputs)

        pred = (output.data >= 0.5).view(-1, 1)
        truth = (labels.data >= 0.5).view(-1, 1)
        num_correct += (pred == truth).sum()
        true_positive += ((pred == truth) & pred ).sum()
        false_positive += ((pred^truth) & pred).sum()
        false_negative += ((pred^truth) & truth).sum()

        if i%20==0:
            print i, num_correct, false_positive, false_negative
        del output
        del inputs
        del labels


    acc = num_correct / len(test_loader.dataset)
    recall = true_positive / (true_positive+false_negative)
    fp = false_positive / len(test_loader.dataset)
    fn = false_negative / len(test_loader.dataset)
    print("recall: %s acc: %s fp: %s fn: %s" %(recall, acc, fp, fn))

