#!/user/bin/env python3
# -*- conding:utf-8 -*-
# @auther Zhang
# @date 2021/8/25
# @file DQN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# Hyper paramters
BATH_SIZE = 32
LR = 0.01  # learning rate
EPSION = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 1000
# 　初始化环境
# CartPole-v0
env = gym.make('CartPole-v0')
# https://blog.csdn.net/weixin_42769131/article/details/114550177
# 可以理解为解除step的次数限制。
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 20)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(20, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)


class DQN():
    def __init__(self):
        self.eval_net = Net()
        self.target_net = Net()
        self.learn_step_count = 0
        self.memory_count = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSION:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value,1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_count % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_count += 1

    def learn(self):
        # target parameter update
        if self.learn_step_count % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_count += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detch()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()
print('\n Collecing experience...')
for i_episode in range(400):
    s = env.reset()
    ep_r = 0
    while True:
        env.render()
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)

        # s
        # Num  Observation                Min                      Max
        # 0     Cart Position             -4.8                    4.8
        # 1     Cart Velocity             -Inf                    Inf
        # 2     Pole Angle                -0.418 rad(-24 deg      0.418 rad (24 deg)
        # 3     Pole Angular Velocity     -Inf                    Inf

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threhold - abs(x)) / env.x_threhold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s,a,r,s_)

        ep_r += r
        if dqn.memory_count >MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('EP: ',i_episode,
                      '| Ep_r: ',round(ep_r,2))
            if done:
                break
            s = s_


