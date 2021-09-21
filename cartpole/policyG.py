#!/user/bin/env python3
# -*- conding:utf-8 -*-
# @auther Zhang
# @date 2021/9/21
# @file policyG.py
# 策略梯度算法
# 2020.5.22
#
# cartpole 的state是一个4维向量，分别是位置，速度，杆子的角度，加速度；action是二维、离散，即向左/右推杆子
# 每一步的reward都是1  游戏的threshold是475
# 这个是MC方法

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


SEED = 543 # 设置种子，以保证复现性
GAMMA = 0.99 # 学习率
RENDER = False # 是否渲染画面
LOG_INTERVAL = 10 # 每隔10轮在控制台输出相关信息
HIDDEN_SIZE = 128 # 中间层节点数目

env = gym.make('CartPole-v1')
env.seed(SEED)
torch.manual_seed(SEED)  # 策略梯度算法方差很大，设置seed以保证复现性
print('observation space:', env.observation_space)
print('action space:', env.action_space)


class Policy(nn.Module):
    ##  离散空间采用了 softmax policy 来参数化策略
    def __init__(self):
        super(Policy, self).__init__()
        '''
        dropout的作用是增加网络的泛化能力，可以用在卷积层和全连接层。但是在卷积层一般不用dropout
        https://blog.csdn.net/junbaba_/article/details/105673998
        '''
        self.fc1 = nn.Linear(4, HIDDEN_SIZE)
        self.dropout = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(HIDDEN_SIZE, 2)  # 两种动作

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.fc2(x)
        # ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        return F.softmax(action_scores, dim=1)


policy = Policy()  # 创建策略网络
optimizer = optim.Adam(policy.parameters(), lr=1e-2) # 创建优化器
eps = np.finfo(np.float32).eps.item()  # 非负的最小值，使得归一化时分母不为0


def select_action(state):
    ## 选择动作，这个动作不是根据Q值来选择，而是使用softmax生成的概率来选
    #  不需要epsilon-greedy，因为概率本身就具有随机性
    state = torch.from_numpy(state).float().unsqueeze(0)
    # print(state.shape)   torch.size([1,4])
    probs = policy(state)
    # print(probs)
    # print(probs.log())
    '''
    class torch.distributions.categorical(probs)
    其作用是创建以参数probs为标准的类别分布，样本是来自“0，...，K-1”的整数，K是probs参数的长度。也就是说，按照probs的概率，在相应的位置进行采样，采样返回的是该位置的整数索引。
    如果probs是长度为K的一维列表，则每个元素是对该索引处的类进行采样的相对概率。
    如果probs是二维的，它被视为一批概率向量
    '''
    m = Categorical(probs)  # 生成分布
    action = m.sample()  # 从分布中采样
    # print(m.log_prob(action))   # m.log_prob(action)相当于probs.log()[0][action.item()].unsqueeze(0)
    policy.saved_log_probs.append(m.log_prob(action))  # 取对数似然 logπ(s,a),这里就等于action的one-hot*action的概率
    return action.item()  # 返回一个元素值

def learn():
    # 一个ep更新结束后,需要更新策略
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:  # 从后到前计算累计reward
        R = r + GAMMA * R  # 给予discount  # 倒序计算累计期望
        returns.insert(0, R)  # 将R插入到指定的位置0处
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)  # 归一化
    for log_prob, R in zip(policy.saved_log_probs, returns):  # 选取动作的概率*reward
        policy_loss.append(-log_prob * R)  # 损失函数为交叉熵
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()  # 求和
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]  # 清空episode 数据
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in range(1000):  # 采集（训练）最多1000个序列
        state, ep_reward = env.reset(), 0  # ep_reward表示每个episode中的reward
        # print(state.shape)
        for t in range(1, 1000):
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if RENDER:
                env.render()  # 刷新当前环境
            policy.rewards.append(reward)
            ep_reward += reward  # 当前一次ep的总reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        learn()  # 根据运行的结果更新策略
        if i_episode % LOG_INTERVAL  == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:  # 大于游戏的最大阈值475时，退出游戏,reward值大于阈值
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()