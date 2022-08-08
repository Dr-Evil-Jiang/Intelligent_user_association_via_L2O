import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


def plot_via_MA(data: np.ndarray, window_size: int):
    """
    Function: smooth the data pts via moving average.
    :param window_size: window size of moving average.
    :param data: ndarray in shape [x, ]
    :return: data
    """
    data = pd.DataFrame(data.flatten()).rolling(window_size).mean()
    return data


def plot_unsupervised_learning_results(path='./results/unsupervised_learning'):
    """
    Function: plot the test results of the unsupervised learning
    :param path: file path where the results are saved.
    :return:
    """
    window_size = 500
    os.makedirs(path, exist_ok=True)
    result_save_path = path
    optimal = plot_via_MA(np.load(os.path.join(result_save_path, 'optimal.npy')), window_size)[window_size-1:]
    random_res = plot_via_MA(np.load(os.path.join(result_save_path, 'random_res.npy')), window_size)[window_size-1:]
    dnn_res = plot_via_MA(np.load(os.path.join(result_save_path, 'dnn_res.npy')), window_size)[window_size-1:]

    total_epochs = dnn_res.shape[0]
    sns.set()
    epochs = np.arange(0, total_epochs, step=1)
    plt.figure(figsize=(30, 3))
    plt.plot(epochs, optimal[0], label="optimal")
    plt.plot(epochs, random_res[0], label="random policy")
    plt.plot(epochs, dnn_res[0], label="unsupervised learning")
    plt.ylabel("Achieved throughput")
    plt.xlabel("Index of testing epochs")
    plt.title("Result")
    plt.legend()
    plt.show()


def plot_supervised_learning_results(path='./results/supervised_learning'):
    """
    Function: plot the test results of the unsupervised learning
    :param path: file path where the results are saved.
    :return:
    """
    window_size = 500
    os.makedirs(path, exist_ok=True)
    result_save_path = path
    optimal = plot_via_MA(np.load(os.path.join(result_save_path, 'optimal.npy')), window_size)[window_size-1:]
    random_res = plot_via_MA(np.load(os.path.join(result_save_path, 'random_res.npy')), window_size)[window_size-1:]
    dnn_res = plot_via_MA(np.load(os.path.join(result_save_path, 'dnn_res.npy')), window_size)[window_size-1:]

    total_epochs = dnn_res.shape[0]
    sns.set()
    epochs = np.arange(0, total_epochs, step=1)
    plt.figure(figsize=(30, 3))
    plt.plot(epochs, optimal[0], label="optimal")
    plt.plot(epochs, random_res[0], label="random policy")
    plt.plot(epochs, dnn_res[0], label="supervised learning")
    plt.ylabel("Achieved throughput")
    plt.xlabel("Index of testing epochs")
    plt.title("Result")
    plt.legend()
    plt.show()


def plot_reinforcement_learning_results(path='./results/dqn'):
    window_size = 500
    os.makedirs(path, exist_ok=True)
    dqn_res = plot_via_MA(np.load(os.path.join(path, 'dqn_res.npy')), window_size)[window_size-1:]
    optimal_res = plot_via_MA(np.load(os.path.join(path, 'optimal_res.npy')), window_size)[window_size-1:]
    total_episodes = dqn_res.shape[0]
    episode_idx = np.arange(0, total_episodes, step=1)
    sns.set()
    plt.figure(figsize=(30, 3))
    plt.plot(episode_idx, optimal_res[0], label="optimal")
    plt.plot(episode_idx, dqn_res[0], label="deep reinforcement learning")
    plt.ylabel("Achieved throughput")
    plt.xlabel("Index of testing epochs")
    plt.title("Result")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_reinforcement_learning_results()