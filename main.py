import os

import numpy as np

from networks import UserAssociationNetSL
from train_nn import *
from DQN import DQNAgent
from utils import *
from datasets import split_train_val_test, split_train_val_test_with_labels

if __name__ == '__main__':
    # Step 1: determine the device on which the model is trained and some parameters:
    # commands:
    OVERWRITE_MODEL = True
    OVERWRITE_DATASET = False
    MODEL = 4  # 1 for unsupervised learning, 2 for reinforcement learning, 3 for supervised learning
    # parameters:
    num_MUs = 10
    num_BDs = 10
    num_batches = 1000  # 1_000_000
    train_batch_size = 1000
    test_batch_size = 500
    max_epochs = 100
    # file paths:
    model_name = 'DNN'
    model_path = './model/dnn_solution'
    data_path = './data'
    result_save_path = './results'

    # Using device:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Step 2: get the train and the test datasets.
    if OVERWRITE_DATASET:
        G, h, f = generate_dataset(num_BDs, num_MUs, num_channel_realizations=num_batches, region=(100, 100))
        np.save(os.path.join(data_path, 'G.npy'), G)
        np.save(os.path.join(data_path, 'h.npy'), h)
        np.save(os.path.join(data_path, 'f.npy'), f)

    if MODEL == 1:
        G, h, f = np.load(os.path.join(data_path, 'G.npy')), \
                  np.load(os.path.join(data_path, 'h.npy')), np.load(os.path.join(data_path, 'f.npy'))
        train_set, val_set, test_set = split_train_val_test(G, f, train_ratio=0.8)
        net = UserAssociationNetUSL(input_dims=num_BDs, output_dims=num_BDs).to(device)
        net = train_unsupervised_learning(net, train_set=train_set, val_set=val_set, device=device,
                                          batch_size=train_batch_size, max_epochs=max_epochs) if OVERWRITE_MODEL \
            else load_model(model_path, model_name)
        test_loader = data.DataLoader(test_set, batch_size=500, pin_memory=True, shuffle=False, drop_last=True)
        test_unsupervised_learning(net, test_set=test_set, save_res=True, plot=True)

    elif MODEL == 2:
        G, h, f = np.load(os.path.join(data_path, 'G.npy')), \
                  np.load(os.path.join(data_path, 'h.npy')), np.load(os.path.join(data_path, 'f.npy'))
        train_set, val_set, test_set = split_train_val_test(G, f, train_ratio=0.8)
        DQN_Agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.001,
                             state_dims=num_BDs + 1, num_actions=num_BDs, action_dims=1)
        train_dqn(DQN_Agent, train_set)
        test_dqn(DQN_Agent, test_set)

    elif MODEL == 3:
        G, h, f = np.load(os.path.join(data_path, 'G.npy')), \
                  np.load(os.path.join(data_path, 'h.npy')), np.load(os.path.join(data_path, 'f.npy'))
        train_set, val_set, test_set = split_train_val_test(G, f, train_ratio=0.8)
        DQN_Agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.001,
                             state_dims=num_BDs, num_actions=num_BDs, action_dims=1,
                             model_name='dqn_solution_with_actions')
        train_dqn(DQN_Agent, train_set, include_actions=False)
        test_dqn(DQN_Agent, test_set, include_actions=False)

    elif MODEL == 4:
        G, h, f = np.load(os.path.join(data_path, 'G.npy')), \
                  np.load(os.path.join(data_path, 'h.npy')), np.load(os.path.join(data_path, 'f.npy'))
        train_set, val_set, test_set = split_train_val_test_with_labels(G, f, train_ratio=0.8)
        net = UserAssociationNetSL(input_dims=num_BDs, output_dims=num_BDs).to(device)
        net = train_supervised_learning(net, train_set=train_set, val_set=val_set, device=device,
                                        batch_size=train_batch_size, max_epochs=max_epochs) if OVERWRITE_MODEL \
            else load_model('./model/supervised_solution', 'supervised_net')
        test_loader = data.DataLoader(test_set, batch_size=500, pin_memory=True, shuffle=False, drop_last=True)
        test_supervised_learning(net, test_set=test_set, save_res=True, plot=True)

