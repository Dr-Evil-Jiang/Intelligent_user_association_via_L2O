import os
import sys
import time
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import DeepQNetwork


class ReplayMemory:
    def __init__(self, max_size, state_dims):
        self.max_size = max_size
        self.state_dims = state_dims
        self.memory_cntr = 0
        self.curr_states = np.zeros((max_size, state_dims), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int64)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dims), dtype=np.float32)
        self.is_terminated = np.zeros(max_size, dtype=np.bool)

    def push_transition(self, curr_state, action, reward, next_state, is_terminated):
        """
        Function: this function is to store the transition.
        :param curr_state: current state, np.ndarray in shape [state_dims]
        :param action: action, np.ndarray in shape [action_dims]
        :param reward: scalar float
        :param next_state: next state, np.ndarray in shape [state_dims]
        :param is_terminated: whether the frame is terminated, bool
        """
        index = self.memory_cntr % self.max_size
        self.curr_states[index] = curr_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.is_terminated[index] = is_terminated
        self.memory_cntr += 1

    def sample_memory(self, batch_size):
        max_mem = min(self.memory_cntr, self.max_size)
        batch_idx = np.random.choice(max_mem, batch_size, replace=False)
        # assert batch_size < self.max_size, f'The input batch_size must be no larger than the size of memory pool!'
        # batch_idx = np.random.randint(low=0, high=self.max_size, size=(batch_size, ))
        curr_state_batch = self.curr_states[batch_idx]
        action_batch = self.actions[batch_idx]
        reward_batch = self.rewards[batch_idx]
        next_state_batch = self.next_states[batch_idx]
        is_terminated_batch = self.is_terminated[batch_idx]
        return curr_state_batch, action_batch, reward_batch, next_state_batch, is_terminated_batch


class DQNAgent:
    def __init__(self, gamma, epsilon, state_dims, action_dims, num_actions, memory_size=50000, batch_size=32,
                 update_every=10000, lr=0.001, eps_dec=5e-5, eps_min=0.1, device='cpu'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.num_actions = num_actions
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.device = device
        self.total_step = 0
        self.memory = ReplayMemory(memory_size, state_dims)

        self.q_eval = DeepQNetwork(state_dims, num_actions).to(device)  # Q eval
        self.q_target = DeepQNetwork(state_dims, num_actions).to(device)  # Q target
        self.q_eval_optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=lr)
        self.loss_module = nn.MSELoss()

    def memorize(self, curr_state, action, reward, next_state, is_terminated):
        self.memory.push_transition(curr_state, action, reward, next_state, is_terminated)

    def get_minibatch(self):
        curr_state_batch, action_batch, reward_batch, next_state_batch, is_terminated_batch = self.memory.sample_memory(
            batch_size=self.batch_size
        )
        curr_state_batch = torch.from_numpy(curr_state_batch).float().to(self.device)
        action_batch = torch.from_numpy(action_batch).float().to(self.device)
        reward_batch = torch.from_numpy(reward_batch).float().to(self.device)
        next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device)
        is_terminated_batch = torch.from_numpy(is_terminated_batch).to(self.device)

        return curr_state_batch, action_batch, reward_batch, next_state_batch, is_terminated_batch

    def take_action(self, observation):
        # observation is in the identical shape with the state_dims
        observation = torch.from_numpy(observation).float().to(self.device)
        if np.random.random() > self.epsilon:  # exploitation
            actions = self.q_eval(observation)
            action = torch.argmax(actions).item()
        else:  # exploration
            action = torch.randint(low=0, high=self.num_actions, size=(1,)).item()
        return action

    def update_network(self):
        if self.total_step % self.update_every == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.memory_cntr < self.batch_size:
            return

        self.update_network()
        curr_state_batch, action_batch, reward_batch, next_state_batch, is_terminated_batch = self.get_minibatch()

        # obtain the predicated Q-value for each of the executed actions
        indices = np.arange(self.batch_size)
        q_pred = self.q_eval(curr_state_batch)[indices, action_batch.long()]
        q_next = self.q_target(next_state_batch)
        q_eval = self.q_eval(next_state_batch)

        max_actions = torch.argmax(q_eval, dim=1)
        q_target = reward_batch + self.gamma * q_next[indices, max_actions] * (1 - is_terminated_batch.to(torch.int))

        q_target = reward_batch + self.gamma * q_target
        self.q_eval_optimizer.zero_grad()
        loss = self.loss_module(q_target, q_pred)
        loss.backward()
        self.q_eval_optimizer.step()

        self.total_step += 1
        self.decrement_epsilon()