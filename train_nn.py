import os
import sys
import time
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from IPython.core.debugger import set_trace
from tqdm.notebook import tqdm
from numba import njit
import matplotlib.pyplot as plt
import seaborn as sns
import json
from utils import *
from networks import *


def train_nn(net, train_set, val_set, device, batch_size=256, max_epochs=100, patience=7, path='./model/dnn_solution',
             model_name='DNN'):
    os.makedirs(path, exist_ok=True)
    val_score = []
    best_val_epoch = -1
    net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=1e-4)
    loss_module = UserAssociationLoss()
    train_loader_local = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                                         pin_memory=True)
    val_loader_local = data.DataLoader(val_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True)

    for epoch in range(max_epochs):
        # Training:
        net.train()
        epoch_loss = 0.0
        for G_batch, f_batch, t_batch in tqdm(train_loader_local, desc=f'Epoch {epoch + 1}', leave=False):
            G_batch, f_batch, t_batch = G_batch.to(device), f_batch.to(device), t_batch.to(device)
            optimizer.zero_grad()
            # channel_gains = torch.abs(G_batch[t_batch.squeeze_(
            #     dim=-1)] * torch.squeeze(f_batch, dim=-1)).float()  # Tensor in shape [batch_size, num_BDs]
            channel_gains = torch.abs(G_batch[t_batch.squeeze_(
                dim=-1)]).float()  # Tensor in shape [batch_size, num_BDs]

            as_vect = net.forward(channel_gains)
            loss = loss_module(G_batch, f_batch, t_batch, as_vect)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch: {epoch} Loss: {epoch_loss / len(train_loader_local)}')
        # Testing:
        gap_avg = val_model(net, val_loader=val_loader_local, device=device)
        val_score.append(gap_avg)
        if len(val_score) == 1 or gap_avg > val_score[best_val_epoch]:
            print('\t (New best performance, saving model...)')
            save_model(net, model_file=path, model_name=model_name)
            best_val_epoch = epoch
        elif best_val_epoch <= epoch - patience:
            print(f'Early stopping due to no improvement over the last {patience} epochs!')
            break

    # plot a curve:
    sns.set()
    plt.plot([i for i in range(1, len(val_score) + 1)], val_score)
    plt.xlabel("Epochs")
    plt.ylabel("Validation accuracy")
    plt.title(f"Validation performance of {model_name}")
    plt.show()
    plt.close()
    return net


def val_model(net, val_loader, device):
    net.eval()
    gap, count = 0.0, 0
    for G_batch, f_batch, t_batch, optimal_throughput, optimal_policy in val_loader:
        G_batch, f_batch, t_batch, optimal_throughput, optimal_policy = G_batch.to(device), f_batch.to(
            device), t_batch.to(device), optimal_throughput.to(device), optimal_policy.to(device)
        with torch.no_grad():
            channel_gains = torch.abs(G_batch[t_batch.squeeze_(dim=-1)]).float()
            as_vect_test = net.forward(channel_gains)
            achieved_throughput = calculate_throughput(G_batch, f_batch, t_batch, as_vect_test)
            count += as_vect_test.shape[0]
            gap += torch.abs(achieved_throughput.sum().item() - optimal_throughput.sum())
    gap_avg = gap / count
    print(f'The testing gap is {gap_avg}')
    return gap_avg


