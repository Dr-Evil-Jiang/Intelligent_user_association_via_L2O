import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import plot_via_MA

if __name__ == '__main__':
    window_size = 500
    result_save_path = './results'
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
