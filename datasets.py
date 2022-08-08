import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from tqdm import tqdm
from utils import calculate_throughput


class ChannelInfoDatasetTrain(data.Dataset):
    def __init__(self, G, f):
        """
        Function:
            To generate the training data points.
        Params:
            G: channel matrix from BDs to MUs, ndarray in shape [num_batches, M, N]
            f: channel vector from BS to BDs, ndarray in shape [num_batches, N, 1]
        """
        self.num_batches, self.num_MUs, self.num_BDs = G.shape
        self.G = torch.from_numpy(G)
        self.f = torch.from_numpy(f)
        self.t = self._get_TDMA_scheduler()

    def _get_TDMA_scheduler(self):
        """
        This function is to get one-hot vectors from TDMA scheduler of MUs.
        """
        t = torch.eye(self.num_MUs)  # all the possible scheduling policy.
        t = t[torch.randint(0, self.num_MUs, size=(self.num_batches,))]
        return t.to(torch.bool).view(-1, self.num_MUs, 1)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        """
        Indexing function:
        """
        return self.G[idx], self.f[idx], self.t[idx]


class ChannelInfoDatasetTest(ChannelInfoDatasetTrain):
    def __init__(self, G, f):
        """
        Function:
            This class is utilized to generate optimal data point
        Params:
            G: channel matrix from BDs to MUs, ndarray in shape [num_batches, M, N]
            f: channel vector from BS to BDs, ndarray in shape [num_batches, N, 1]
        """
        super().__init__(G, f)
        self.optimal_throughput, self.optimal_policy = self._get_optimal_policy()

    def _get_optimal_policy(self):
        """
        G: channel matrix from BDs to MUs, Tensor in shape [batch_size, M, N]
        f: channel vector from BS to BDs, Tensor in shape [batch_size, N, 1]
        t: TDMA scheduler indicating which MU is currently transmitting, Tensor [bool], in shape [batch_size, M, 1]
        """
        policy_arr = torch.eye(self.num_BDs, dtype=torch.float32)  # all possible actions
        optimal_throughput = torch.zeros((self.num_batches, self.num_BDs))

        for idx in tqdm(range(policy_arr.shape[0]), leave=False,
                        desc=f'finding optimal user scheduling vector via exhaustive search...'):
            policy_arr_temp = torch.stack([policy_arr[idx] for _ in range(self.num_batches)], dim=0)
            received_signal = calculate_throughput(self.G, self.f, self.t, policy_arr_temp)
            optimal_throughput[:, idx] = received_signal

        # print(optimal_throughput.shape, optimal_policy.shape)
        optimal_policy = policy_arr[torch.argmax(optimal_throughput, dim=1)]
        optimal_throughput, _ = torch.max(optimal_throughput, dim=1, keepdim=True)

        return optimal_throughput, optimal_policy

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        """
        Indexing function:
        """
        return self.G[idx], self.f[idx], self.t[idx], self.optimal_throughput[idx], self.optimal_policy[idx]


def split_train_val_test(G, f, train_ratio=0.8):
    """
    Function: this function is to split the train set, validation set, and the test set
    :param G: ndarray in shape [batch_size, M, N]
    :param f: ndarray in shape [batch_size, M, 1]
    :param train_ratio: float the ratio between the number of train batches to the number of total batches
    :return:
    """
    total_batch = G.shape[0]
    train_size = int(train_ratio * total_batch)
    # Train set:
    train_set = ChannelInfoDatasetTrain(G[:train_size], f[:train_size])
    # Validation set:
    val_batch_idx = np.random.randint(0, train_size, size=(int(train_size * 0.3)))
    val_set = ChannelInfoDatasetTest(G[val_batch_idx], f[val_batch_idx])
    # Test set:
    test_set = ChannelInfoDatasetTest(G[train_size:], f[train_size:])
    return train_set, val_set, test_set


def split_train_val_test_with_labels(G, f, train_ratio=0.8):
    total_batch = G.shape[0]
    train_size = int(train_ratio * total_batch)
    # Train set:
    train_set = ChannelInfoDatasetTest(G[:train_size], f[:train_size])
    # Validation set:
    val_batch_idx = np.random.randint(0, train_size, size=(int(train_size * 0.3)))
    val_set = ChannelInfoDatasetTest(G[val_batch_idx], f[val_batch_idx])
    # Test set:
    test_set = ChannelInfoDatasetTest(G[train_size:], f[train_size:])
    return train_set, val_set, test_set
