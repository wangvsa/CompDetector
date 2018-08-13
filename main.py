import numpy as np
import time, sys
import glob, os, math, random, argparse
import torch, torchvision
import torch.nn as nn
import alex
from alex import FlashDataset, FlashNet

BATCH_SIZE = 10
USE_GPU = True

def load_model(model_file):
    model = None
    if os.path.isfile(model_file):
        print "Load existing model"
        if not USE_GPU:    # Load model to CPU (trained on multiple GPUs),
            model = torch.load(model_file, map_location="cpu").module
        else:
            model = torch.load(model_file)
    else:
        model = FlashNet().double()
        if torch.cuda.is_available() and USE_GPU:
            print "Have CUDA!!!"
            model = model.cuda()
        if torch.cuda.device_count() > 1 and USE_GPU:
            print "More than one GPU card!!!"
            model = nn.DataParallel(model)
    print model
    return model

if __name__ == "__main__":

    t1 = time.time()
    model_file = "./sod.model"
    model = load_model(model_file)
    t2 = time.time()
    print("loading time: ", t2-t1)

    # Training
    trainset = FlashDataset(sys.argv[1])
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    alex.training(model, train_loader, epochs=3, use_gpu=USE_GPU)
    torch.save(model, model_file)

    # Testing
    alex.evaluating(model, train_loader, use_gpu=USE_GPU)
