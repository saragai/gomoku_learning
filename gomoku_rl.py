# -*- coding: utf-8 -*-
from gomoku.envs.gomoku_env import GomokuEnv
import gym
import numpy as np
from collections import namedtuple
from itertools import count

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision
from torchvision import datasets, models, transforms
from torch.distributions import Categorical

from Network import Policy

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

model = Policy()
optimizer = optim.Adam(model.parameters(), lr=0.01)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    returns = []
    for r in model.rewards[::-1]:
        R = r + 0.9 * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    # env = gym.make('gomoku-v0')
    # env.set_reward(win=10)

    env = gym.make("CartPole-v0")
    env.seed(0)
    torch.manual_seed(0)

    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):
            action = select_action(state)
            state, reward, done, _ = env.step(action)

            if i_episode % 100 == 0:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        finish_episode()

        running_reward = 0.05 * ep_reward + 0.95 * running_reward
        if i_episode % 100 == 0:
            print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}')

    env.close()


if __name__ == '__main__':
    main()
