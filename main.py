import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from networks import *
from train_nn import train_nn
from datasets import *
from tqdm import tqdm

if __name__ == '__main__':
    # Step 1: determine the device on which the model is trained and some parameters:
    # commands:
    OVERWRITE_MODEL = True
    OVERWRITE_DATASET = True
    # parameters:
    num_MUs = 10
    num_BDs = 10
    num_batches = 1000_000
    train_batch_size = 500
    test_batch_size = 500
    max_epochs = 100
    # file paths:
    model_name = 'DNN'
    model_path = './model/dnn_solution'
    train_data_path = './data/train_data.tar'
    test_data_path = './data/test_data.tar'
    result_save_path = './results'

    # Using device:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Step 2: get the train and the test datasets.
    G, h, f = generate_dataset(num_BDs, num_MUs, num_channel_realizations=num_batches, region=(100, 100))
    train_set = ChannelInfoDatasetTrain(G, f)
    val_set = ChannelInfoDatasetTest(G, f)

    # Step 3: create the NN model to be trained and train the model.
    net = UserAssociationNet(input_dims=num_BDs, output_dims=num_BDs).to(device)
    net = train_nn(net, train_set=train_set, val_set=val_set, device=device,
                   batch_size=train_batch_size, max_epochs=max_epochs) if OVERWRITE_MODEL \
        else load_model(model_path, model_name)

    # Step 4: test the model
    test_loader = data.DataLoader(val_set, batch_size=500, pin_memory=True, shuffle=False, drop_last=True)
    optimal = torch.Tensor()
    random_res = torch.Tensor()
    dnn_res = torch.Tensor()
    for G_batch, f_batch, t_batch, optimal_throughput, optimal_policy in tqdm(test_loader,
                                                                              desc='Testing the model performance'):
        G_batch, f_batch, t_batch, optimal_throughput, optimal_policy = G_batch.to(device), f_batch.to(
            device), t_batch.to(device), optimal_throughput.to(device), optimal_policy.to(device)
        with torch.no_grad():
            channel_gains = torch.abs(G_batch[t_batch.squeeze_(dim=-1)]).float()
            as_vect_test = net.forward(channel_gains)
            # as_vect_test = output2onehot(as_vect_test)
            achieved_throughput = calculate_throughput(G_batch, f_batch, t_batch, as_vect_test)
        optimal = torch.cat([optimal, optimal_throughput.cpu().view(-1)], dim=0)
        dnn_res = torch.cat([dnn_res, achieved_throughput.cpu().view(-1)], dim=0)
        random_policy = (torch.eye(num_BDs, dtype=torch.float32))[
            torch.randint(high=num_BDs, size=(achieved_throughput.shape[0],))].to(device)
        achieved_throughput = calculate_throughput(G_batch, f_batch, t_batch, random_policy)
        random_res = torch.cat([random_res, achieved_throughput.cpu().view(-1)], dim=0)

    # Step 5: Save and plot the results.
    os.makedirs(result_save_path, exist_ok=True)
    percentage = 0.0001
    optimal = optimal.cpu().numpy()
    random_res = random_res.cpu().numpy()
    dnn_res = dnn_res.cpu().numpy()
    np.save(os.path.join(result_save_path, 'optimal.npy'), optimal)
    np.save(os.path.join(result_save_path, 'random_res.npy'), random_res)
    np.save(os.path.join(result_save_path, 'dnn_res.npy'), dnn_res)

    sns.set()
    total_epochs = dnn_res.shape[0]
    plotted_epochs = int(percentage * total_epochs)
    epochs = np.arange(0, total_epochs, step=1)
    plt.figure(figsize=(18, 3))
    plt.plot(epochs[:plotted_epochs], optimal[:plotted_epochs], label="optimal")
    plt.plot(epochs[:plotted_epochs], random_res[:plotted_epochs], label="random policy")
    plt.plot(epochs[:plotted_epochs], dnn_res[:plotted_epochs], label="unsupervised learning")
    plt.ylabel("Achieved throughput")
    plt.xlabel("Index of testing epochs")
    plt.title("Result")
    plt.legend()
    plt.show()
