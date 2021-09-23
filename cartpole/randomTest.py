#!/user/bin/env python3
# -*- conding:utf-8 -*-
# @auther Zhang
# @date 2021/8/19
# @file
import gym
import numpy as np

env = gym.make('CartPole-v0')

max_number_of_steps = 200  # 每一场游戏的最高得分
# ---------获胜的条件是最近100场平均得分高于195-------------
goal_average_steps = 195
num_consecutive_iterations = 100
# ----------------------------------------------------------
num_episodes = 1000  # 共进行1000场游戏
last_time_steps = np.zeros(num_consecutive_iterations)  # 只存储最近100场的得分（可以理解为是一个容量为100的栈）

# 重复进行一场场的游戏
for episode in range(num_episodes):
    observation = env.reset()  # 初始化本场游戏的环境
    episode_reward = 0  # 初始化本场游戏的得分
    # 一场游戏分为一个个时间步
    for t in range(max_number_of_steps):
        env.render()  # 更新并渲染游戏画面
        action = np.random.choice([0, 1])  # 随机决定小车运动的方向
        observation, reward, done, info = env.step(action)  # 获取本次行动的反馈结果
        episode_reward += reward
        if done:
            print('%d Episode finished after %f time steps / mean %f' % (episode, t + 1, last_time_steps.mean()))
            last_time_steps = np.hstack((last_time_steps[1:], [episode_reward]))  # 更新最近100场游戏的得分stack
            break
    # 如果最近100场平均得分高于195
    if (last_time_steps.mean() >= goal_average_steps):
        print('Episode %d train agent successfuly!' % episode)
        break

print('Failed!')
