import os

import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from utils import calculate_throughput, save_model, \
    plot_unsupervised_learning_results, plot_reinforcement_learning_results
from DQN import DQNAgent
from networks import *


def train_unsupervised_learning(net, train_set, val_set, device, batch_size=256, max_epochs=100, patience=7,
                                path='./model/dnn_solution', model_name='DNN'):
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
        # validating:
        gap_avg = val_unsupervised_learning(net, val_loader=val_loader_local, device=device)
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


def val_unsupervised_learning(net, val_loader, device):
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
    print(f'The validating gap to the optimal results is {gap_avg}')
    return gap_avg


def test_unsupervised_learning(net, test_set, save_res=True, path='./results/unsupervised_learning', plot=False):
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


def train_dqn(DQN_Agent, train_dataset, include_actions=True):
    """
    This function is utilized to train DQN
    :param include_actions: whether to include action to guide the learning process
    :param DQN_Agent: an obj. of class DQNAgent
    :param train_dataset: pytorch dataset
    """
    train_loader = data.DataLoader(train_dataset, shuffle=False, batch_size=1, drop_last=False)
    G_batch, f_batch, t_batch = train_dataset[0]
    G_batch.unsqueeze_(0)
    f_batch.unsqueeze_(0)
    t_batch.unsqueeze_(0)
    action = torch.randint(0, DQN_Agent.num_actions, (1, ))
    observation = torch.abs(G_batch[t_batch.squeeze_(dim=-1)]).float()
    if include_actions:
        observation = torch.cat((observation, torch.Tensor([[action]])), dim=-1)
    reward_trace = []
    best_reward = -np.inf
    for step in tqdm(range(1, len(train_loader)),
                     desc=f'Train DQN'):
        action = DQN_Agent.take_action(observation.numpy())  # the index of action and needs to be converted to one-hot.
        as_vect = torch.zeros((1, DQN_Agent.num_actions),
                              dtype=torch.float32).scatter_(1, torch.Tensor([[action]]).long(), 1.0)
        reward = calculate_throughput(G_batch, f_batch, t_batch, as_vect).item()
        done = 1  # Since this game has no episode.
        G_batch, f_batch, t_batch = train_dataset[step]
        G_batch.unsqueeze_(0)
        f_batch.unsqueeze_(0)
        t_batch.unsqueeze_(0)
        observation_ = torch.abs(G_batch[t_batch.squeeze_(dim=-1)]).float()
        if include_actions:
            observation_ = torch.cat((observation_, torch.Tensor([[action]])), dim=-1)
        DQN_Agent.memorize(observation.numpy(), action, reward, observation_.numpy(), int(done))
        DQN_Agent.learn()
        observation = observation_
        if reward > best_reward:
            best_reward = reward
            DQN_Agent.save_model()
        reward_trace.append(reward)


def test_dqn(DQN_Agent, test_dataset, save_results=True, path='./results/dqn', plot=True, include_actions=True):
    """
    This function is utilized to test DQN with obtained weights and bias
    :param DQN_Agent: an obj. of class DQNAgent
    :param test_dataset: pytorch dataset
    :param save_results: whether to save the results
    """
    DQN_Agent.load_model()
    print(f'(The trained model has been loaded, successfully!)')
    dqn_res = np.zeros(len(test_dataset))
    optimal_res = np.zeros(len(test_dataset))
    for step in tqdm(range(len(test_dataset)), desc='Testing the DQN model'):
        G_batch, f_batch, t_batch, optimal_throughput, _ = test_dataset[step]
        G_batch.unsqueeze_(0)
        f_batch.unsqueeze_(0)
        t_batch.unsqueeze_(0)
        observation = torch.abs(G_batch[t_batch.squeeze_(dim=-1)]).float()
        if include_actions:
            observation = torch.cat((observation, torch.Tensor([[torch.randint(0, DQN_Agent.num_actions, (1, ))]]))
                                    , dim=-1)
        action = DQN_Agent.take_action(observation.numpy())
        as_vect = torch.zeros((1, DQN_Agent.num_actions),
                              dtype=torch.float32).scatter_(1, torch.Tensor([[action]]).long(), 1.0)
        reward = calculate_throughput(G_batch, f_batch, t_batch, as_vect)
        dqn_res[step] = reward.item()
        optimal_res[step] = optimal_throughput.item()
    if save_results:
        if include_actions:
            os.makedirs(path, exist_ok=True)
            np.save(os.path.join(path, 'dqn_res_actions.npy'), dqn_res)
            np.save(os.path.join(path, 'optimal_res.npy'), optimal_res)
        else:
            os.makedirs(path, exist_ok=True)
            np.save(os.path.join(path, 'dqn_res_no_actions.npy'), dqn_res)
            np.save(os.path.join(path, 'optimal_res.npy'), optimal_res)
    if plot:
        plot_reinforcement_learning_results(path)


def train_supervised_learning(net, train_set, val_set, device, batch_size=256, max_epochs=100, patience=7,
                              path='./model/supervised_solution', model_name='supervised_net'):
    os.makedirs(path, exist_ok=True)
    val_score = []
    best_val_epoch = -1
    net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=1e-4)
    loss_module = nn.CrossEntropyLoss()
    train_loader_local = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                                         pin_memory=True)
    val_loader_local = data.DataLoader(val_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True)

    for epoch in range(max_epochs):
        # Training loop
        net.train()
        epoch_loss = 0.0
        for G_batch, f_batch, t_batch, optimal_throughput, optimal_policy in \
                tqdm(train_loader_local, desc=f'Epoch {epoch + 1}', leave=False):
            G_batch, f_batch, t_batch, optimal_throughput, optimal_policy = \
                G_batch.to(device), f_batch.to(device), t_batch.to(device), \
                optimal_throughput.to(device), optimal_policy.to(device)
            channel_gains = torch.abs(G_batch[t_batch.squeeze_(
                dim=-1)]).float()  # Tensor in shape [batch_size, num_BDs]
            as_vect = net.forward(channel_gains)
            optimizer.zero_grad()
            loss = loss_module(as_vect, optimal_policy)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch: {epoch} Loss: {epoch_loss / len(train_loader_local)}')

        # Validating
        score = val_supervised_learning(net, val_loader_local, device)
        val_score.append(score)
        if len(val_score) == 1 or score > val_score[best_val_epoch]:
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


def val_supervised_learning(net, val_loader, device):
    net.eval()
    true_preds, cntr = 0.0, 0
    for G_batch, f_batch, t_batch, optimal_throughput, optimal_policy in val_loader:
        G_batch, f_batch, t_batch, optimal_throughput, optimal_policy = G_batch.to(device), f_batch.to(
            device), t_batch.to(device), optimal_throughput.to(device), optimal_policy.to(device)
        with torch.no_grad():
            channel_gains = torch.abs(G_batch[t_batch.squeeze_(dim=-1)]).float()
            as_vect_test = net.forward(channel_gains)
            as_vect_test = as_vect_test.argmax(dim=1)
            optimal_policy = optimal_policy.argmax(dim=1)
            true_preds += (as_vect_test == optimal_policy).sum().item()
            cntr += optimal_policy.shape[0]
    val_score = true_preds / cntr
    print(f'The validation hit rate is {val_score}')
    return val_score


def test_supervised_learning(net, test_set, save_res=True, path='./results/supervised_learning', plot=False):
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
        torch.randint(high=as_vect_test.shape[1], size=(achieved_throughput.shape[0], ))]
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





