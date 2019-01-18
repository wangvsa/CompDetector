import torch, torchvision
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import glob
import os
import math
import random
import argparse
from create_dataset import get_flip_error

BATCH_SIZE = 256
variables = ['dens']
NX, NY, NZ = 16, 16, 16
CONV_INPUT_SHAPE = (len(variables), NX, NY, NZ)

WINDOWS_PER_FRAME = 512


class FlashDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, detection=False):
        self.data_files= []
        self.targets = []

        # Add error data
        error_data_files = glob.glob(data_dir+"/*/error/*")
        #error_data_files.sort()
        if detection:   # when doing detection, only one window in a frame has an error
            error_targets = np.zeros((len(error_data_files), 1))
            for i in range(len(error_data_files)/WINDOWS_PER_FRAME):
                idx = random.randint(0, WINDOWS_PER_FRAME-1)
                error_targets[i * WINDOWS_PER_FRAME + idx][0] = 1
            self.targets.append(error_targets)
        else:
            self.targets.append(np.ones((len(error_data_files), 1)))

        # Add clean data
        clean_data_files = glob.glob(data_dir+"/*/clean/*")
        clean_data_files.sort()
        self.targets.append(np.zeros((len(clean_data_files), 1)))


        self.data_files = error_data_files+clean_data_files
        self.targets = np.vstack(self.targets)
        self.targets = torch.from_numpy(self.targets)
        print "data size:", len(self.data_files), ", targets size: ", self.targets.size()

    def __getitem__(self, index):
        return self.read_binary(index, self.data_files[index]), self.targets[index]
    def __len__(self):
        return self.targets.size(0)
    def read_binary(self, index, filename):
        f = filename
        if "error" in f:
            f = filename.replace("error", "clean")
            #data = np.fromfile(f, dtype=np.double).reshape(NX, NY, NZ, 3)[:,:,:,1:2]
            data = np.load(f)
            data = data.reshape(1, NX, NY, NZ)
            if self.targets[index]:     # when doing detection, only one window in a frame has an error
                x, y, z = random.randint(1, NX-2), random.randint(1, NY-2), random.randint(1, NZ-2)
                data[0,x,y,z] = get_flip_error(data[0,x,y,z], 15)
        else:
            #data = np.fromfile(f, dtype=np.double).reshape(NX, NY, NZ, 3)[:,:,:,1:2]
            data = np.load(f)
            data = data.reshape(1, NX, NY, NZ)
        return torch.from_numpy(data)

class FlashNet(nn.Module):
    def __init__(self):
        super(FlashNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, 3),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            #nn.MaxPool3d(2),
            nn.Conv3d(32, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(3),
        )
        conv_output_size = self.get_conv_output_size()
        print "conv output size: ", conv_output_size
        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_output_size, out_features=256, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(model.parameters(), lr = 1e-6, momentum=0.5)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
    loss_func = nn.BCELoss()

    running_loss = 0
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            if torch.cuda.is_available() and use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0 and i != 0:
                print("epoch:%s i:%s loss:%s" %(epoch, i, running_loss/10.0))
                running_loss = 0

def evaluating(model, test_loader, use_gpu=True):
    total_pred = []
    total_truth = []

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

        pred = (output.data >= 0.99).view(-1, 1)
        truth = (labels.data >= 0.5).view(-1, 1)
        num_correct += (pred == truth).sum().item()
        true_positive += ((pred == truth) & pred ).sum().item()
        false_positive += ((pred^truth) & pred).sum().item()
        false_negative += ((pred^truth) & truth).sum().item()

        total_pred.append(pred.cpu().numpy())
        total_truth.append(truth.cpu().numpy())

        if i % 10 == 0:
            print i, num_correct, true_positive, false_positive, false_negative
        del output
        del inputs
        del labels


    print num_correct, true_positive, false_positive, false_negative
    acc = 0 if num_correct == 0 else num_correct / len(test_loader.dataset)
    recall = 0 if true_positive == 0 else true_positive / (true_positive+false_negative)
    fp = 0 if false_positive == 0 else false_positive / len(test_loader.dataset)
    fn = 0 if false_negative == 0 else false_negative / len(test_loader.dataset)
    print("recall: %.4f(%s) acc: %.4f fp: %.4f fn: %.4f" %(recall, (true_positive+false_negative), acc, fp, fn))



    total_pred = np.vstack(total_pred)
    total_truth = np.vstack(total_truth)
    total_pred = total_pred.reshape((total_pred.shape[0]/WINDOWS_PER_FRAME,WINDOWS_PER_FRAME))
    total_truth = total_truth.reshape((total_truth.shape[0]/WINDOWS_PER_FRAME,WINDOWS_PER_FRAME))
    print total_pred.shape, total_truth.shape
    #compute_metrics(np.any(total_pred, axis=1), np.any(total_truth, axis=1))
    compute_metrics(np.sum(total_pred, axis=1) == 1, np.any(total_truth, axis=1))

def compute_metrics(pred_labels, true_labels):
    print pred_labels, true_labels
    clean_samples = np.sum(true_labels == 0) * 1.0
    error_samples = np.sum(true_labels == 1) * 1.0
    total = clean_samples + error_samples * 1.0
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    tp = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    tn = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    fp = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    fn = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    recall, fpr = tp / error_samples, fp / total
    accuracy = (tp + tn) / total
    print 'TP: %s (%i/%i), FP: %s (%i/%i)' %(recall, tp, error_samples, fpr, fp, total)
    print 'ACC: %s, TN: %i, FN: %i' %(accuracy, tn, fn)
