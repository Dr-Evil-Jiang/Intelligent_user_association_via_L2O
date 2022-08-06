import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from tqdm import tqdm
from utils import *


# Networks:
class UserAssociationNet(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dp2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(64, output_dims)
        self.elu = nn.ELU(inplace=True)
        self.gumbel_softmax = F.gumbel_softmax

    def forward(self, x):
        output = self.fc1(x)
        output = self.elu(output)
        output = self.dp1(output)
        output = self.fc2(output)
        output = self.elu(output)
        output = self.dp2(output)
        output = self.fc3(output)
        output = self.gumbel_softmax(output, hard=False) if self.training else self.gumbel_softmax(output, hard=True)
        return output


# Deep Q-Nets:
class DeepQNetwork(nn.Module):
    def __init__(self, state_dims, action_dims):
        super().__init__()
        self.fc1 = nn.Linear(state_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dims)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, state_batch):
        x = self.fc1(state_batch)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# Loss modules:
class UserAssociationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(G, f, t, as_vect):
        """
        Function: forward call
        G: channel matrix from BDs to MUs, Tensor in shape [batch_size, M, N]
        f: channel vector from BS to BDs, Tensor in shape [batch_size, N, 1]
        t: TDMA scheduler indicating which MU is currently transmitting, Tensor one-hot[bool], in shape [batch_size, M, 1]
        as_vect: user association vector via Grumble trick , Tensor in shape [batch_size, N]
        """
        w = 0.90
        loss = - w * torch.sum(calculate_throughput(G, f, t, as_vect)) \
               - (1 - w) * torch.sum(as_vect * torch.log2(as_vect))  # Not good
        # loss = - torch.sum(calculate_throughput(G, f, t, as_vect)) # good performance
        return loss

