import os
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
from plot_fns import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def generate_channel(*args):
    x_dim, y_dim = args
    H = np.sqrt(0.5) * np.random.randn(x_dim, y_dim) + 1j * np.sqrt(0.5) * np.random.randn(x_dim, y_dim)
    return H


def get_BD2MU_dist(BD_locals, MU_locals):
    """
    Function: calculate the distance between the BS_locals and the MU_locals
    params:
        BD_locals: ndarray in shape [N, 2]
        MU_locals: ndarray in shape [M, 2]
    returns:
        BD2MU_dist: ndarray in shape [M, N]
    """
    N, M = BD_locals.shape[0], MU_locals.shape[0]
    BD2MU_dist = np.zeros((M, N))
    for i in range(M):
        for ii in range(N):
            BD2MU_dist[i, ii] = np.linalg.norm(BD_locals[ii] - MU_locals[i])

    return BD2MU_dist


def PL_model(distance_arr):
    """
    Function: evaluate the pass-loss via the inherent PL model
    noting: the returns is precessed by a np.sqrt() operation.
    :params:
        distance_arr: ndarray
    returns:
        pl_arr: ndarray in the identical shape with the input mat.
    """
    # In the formula, carrier freq. should be converted to MHz scale
    # and, at the meantime, the distance is evaluated by KM.
    pl_arr = np.power(10, -0.1 * (32.45 + 20 * np.log10(2.4e3) + 20 * np.log10(1e-3 * distance_arr) - 5))
    return np.sqrt(pl_arr)


def generate_dataset(num_BDs, num_MUs, num_channel_realizations, region=(100, 100)):
    """
    Function: to obtain the
    :param num_BDs: int
    :param num_MUs: int
    :param num_channel_realizations: int
    :param region: the area where BDs and MUs are scattered via random policy, tuple (x_range, y_range)
    :return:
        G：channel between BDs to MUs ndarray in shape [num_channel_realizations, num_MUs, num_BDs]
        h：channel between BS to MUs ndarray in shape [num_channel_realizations, num_MUs, 1]
        f：channel between BS to BDs ndarray in shape [num_channel_realizations, num_BDs, 1]
    """
    BD_locals = np.stack([np.random.randint(-region[0], region[1], size=(num_BDs,)), np.random.randint(-50, 50, size=(num_BDs,))],
                         axis=1)
    MU_locals = np.stack([np.random.randint(-region[0], region[1], size=(num_MUs,)), np.random.randint(-50, 50, size=(num_MUs,))],
                         axis=1)
    BS_local = np.array([0.0, 0.0], ndmin=2)
    # calculating all the distances:
    BS2BD_dist = np.sqrt(
        np.sum(np.power(BD_locals - np.repeat(BS_local, BD_locals.shape[0], axis=0), 2), axis=1, keepdims=True))
    BS2MU_dist = np.sqrt(
        np.sum(np.power(MU_locals - np.repeat(BS_local, MU_locals.shape[0], axis=0), 2), axis=1, keepdims=True))
    BD2MU_dist = get_BD2MU_dist(BD_locals, MU_locals)
    # calculating the large-scale fading components:
    BS2BD_PL = PL_model(BS2BD_dist)
    BS2MU_PL = PL_model(BS2MU_dist)
    BD2MU_PL = PL_model(BD2MU_dist)
    # adding the small-scale fading coefficients:
    G = np.stack([BD2MU_PL*generate_channel(*BD2MU_PL.shape) for _ in range(num_channel_realizations)], axis=0)
    h = np.stack([BS2MU_PL*generate_channel(*BS2MU_PL.shape) for _ in range(num_channel_realizations)], axis=0)
    f = np.stack([BS2BD_PL*generate_channel(*BS2BD_PL.shape) for _ in range(num_channel_realizations)], axis=0)
    return G, h, f


def calculate_throughput(G, f, t, as_vect):
    """
    Function: calculate throughput.
    :param
        G: channel matrix from BDs to MUs, Tensor in shape [batch_size, M, N]
        f: channel vector from BS to BDs, Tensor in shape [batch_size, N, 1]
        t: TDMA scheduler indicating which MU is currently transmitting, Tensor one-hot[bool], in shape [batch_size, M, 1]
        as_vect: user association vector via Grumble trick, Tensor in shape [batch_size, N]
    :returns
        received_signal: achieved throughput for each of the BDs, Tensor in shape [batch_size, N]
    """
    alpha = np.sqrt(0.8)
    sigma = np.sqrt(1e-1 * np.power(10, -11.4))  # sigma**2 = -114 dBm
    P = 10  # Watt, transmit power at BS
    received_signal = torch.abs(G[t.squeeze_(
        dim=-1)]).square_()  # in shape [batch_size, N] to select the current active MU according to TDMA protocol
    received_signal = P * (alpha ** 2) * received_signal.mul(
        torch.abs(torch.einsum('bi, bi -> bi', f.squeeze_(dim=-1), as_vect)).square_())  # element-wise production
    received_signal = torch.log2(1.0 + (torch.sum(received_signal, dim=1) / (sigma ** 2)))
    return received_signal


def save_model(net, model_file, model_name):
    """
    Function: to save the model
    """
    model_name += '.pt'
    assert os.path.isdir(model_file), print(f'There is such a path {model_file}')
    torch.save(net, os.path.join(model_file, model_name))
    print(f'(Model is saved successfully at {os.path.join(model_file, model_name)}!)')


def load_model(model_file, model_name):
    """
    Function: to load the model
    """
    model_name += '.pt'
    assert os.path.isfile(os.path.join(model_file, model_name)), \
        print(f'There is no model file at {os.path.join(model_file, model_name)}')
    net = torch.load(os.path.join(model_file, model_name))
    print('(Model is loaded successfully!)')
    return net


def unsupervised_learning_test(net, test_set, save_res=True, path ='./results/unsupervised_learning', plot=False):
    """
    Function: to evaluate the performance of unsupervised learning
    :param net: pytorch model obj
    :param test_set: pytorch.data.Dataset
    :param save_res: whether to save the results
    :param path: where the results will be saved
    :param plot: bool whether to plot the results
    :return:
    """
    net.eval()
    net.cpu()
    test_loader = data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)
    optimal = torch.Tensor()
    random_res = torch.Tensor()
    dnn_res = torch.Tensor()
    G_batch, f_batch, t_batch, optimal_throughput, _ = next(iter(test_loader))

    with torch.no_grad():
        channel_gains = torch.abs(G_batch[t_batch.squeeze_(dim=-1)]).float()
        as_vect_test = net.forward(channel_gains)
        achieved_throughput = calculate_throughput(G_batch, f_batch, t_batch, as_vect_test)
    optimal = torch.cat([optimal, optimal_throughput.view(-1)], dim=0)
    dnn_res = torch.cat([dnn_res, achieved_throughput.view(-1)], dim=0)
    random_policy = (torch.eye(as_vect_test.shape[1], dtype=torch.float32))[
        torch.randint(high=as_vect_test.shape[1], size=(achieved_throughput.shape[0],))]
    achieved_throughput = calculate_throughput(G_batch, f_batch, t_batch, random_policy)
    random_res = torch.cat([random_res, achieved_throughput.cpu().view(-1)], dim=0)
    if save_res:
        os.makedirs(path, exist_ok=True)
        optimal = optimal.cpu().numpy()
        random_res = random_res.cpu().numpy()
        dnn_res = dnn_res.cpu().numpy()
        np.save(os.path.join(path, 'optimal.npy'), optimal)
        np.save(os.path.join(path, 'random_res.npy'), random_res)
        np.save(os.path.join(path, 'dnn_res.npy'), dnn_res)
    if plot:
        plot_unsupervised_learning_results(path)













