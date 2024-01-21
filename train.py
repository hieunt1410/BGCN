#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time
import os

def train(model, epoch, loader, optim, device, CONFIG, loss_func):
    log_interval = CONFIG['log_interval']
    model.train()
    start = time()
    for i, data in enumerate(loader):
        users_b, bundles = data
        modelout = model(users_b.to(device), bundles.to(device))
        loss = loss_func(modelout, batch_size=loader.batch_size)
        optim.zero_grad()
        loss.backward()
        optim.step()
    print('Train Epoch: {}: time = {:d}s'.format(epoch, int(time()-start)))
    return loss