#!/user/bin/env python3
# -*- conding:utf-8 -*-
# @auther Zhang
# @date 2021/8/2
# @file RandomSelection.py

# https://zhuanlan.zhihu.com/p/112970426
'''
ε-greedy策略，它的核心思想是在游戏中设置一个探索率ε
以ε为概率进行探索，随机选择一个摇臂；
以概率1− 进行利用，即选择当前平均奖励最高的摇臂，其算法流程如下：
1. 建立期望奖励估计表，设定探索率为ε，首先对每个摇臂操作十次，计算奖励数据，作为每个摇臂的期望奖励的初始值
每次游戏中：
    产生服从0-1之间均匀分布的随机数x
        如果x>ε:
            选择当前最大的期望奖励所对应的那个摇臂进行操作
        否则：
            随机选择一个摇臂进行操作
    收集操作反馈的奖励数据，更新期望奖励估计表
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 设定探索率
# 设摇臂个数为5，编号为0到4
k = 5
# 每个摇臂的真实奖励概率
real_reward = [0.2, 0.4, 0.7, 0.5, 0.3]
# 探索率设置为0.1
explorate_rate = 0.1
def epsilon_greedy(N, explorate_rate): # N为游戏次数

    # 初始化各摇臂期望奖励估计
    expect_reward_estimate = [0] * k
    # 初始化各摇臂操作次数
    operation = [0] * k
    # 初始化总奖励
    total_reward = 0
    for i in range(N):
        '''
        numpy.random.uniform()介绍：函数原型： numpy.random.uniform(low,high,size)
        功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
        参数介绍:low: 采样下界，float类型，默认值为0；
        high: 采样上界，float类型，默认值为1；
        size: 输出样本数目，为int或元组(tuple)类型，
        例如，size=(m,n,k), 则输出 m * n * k 个样本，缺省时输出1个值。返回值：ndarray类型，其形状和参数size中描述一致。
        '''
        # 产生一个服从0到1之间均匀分布的随机数
        r = np.random.uniform(size=1)[0]
          # 选择“利用”
        if r > explorate_rate:
            # index() 函数用于从列表中找出某个值第一个匹配项的索引位置。
            # list.index(x[, start[, end]])
            # x-- 查找的对象。
            # start-- 可选，查找的起始位置。
            # end-- 可选，查找的结束位置。
            # 选择当前最大期望奖励所对应的摇臂进行操作
            best_arm =  expect_reward_estimate.index(max(expect_reward_estimate))
        # 选择“探索”
        else:
            # 随机选择摇臂进行操作
            best_arm = np.random.choice(k, size=1)[0]
        # 收集反馈的奖励数据
        best_arm_reward = np.random.binomial(1, real_reward[best_arm], size=1)[0]
        # 更新期望奖励估计
        expect_reward_estimate[best_arm] = (expect_reward_estimate[best_arm] * operation[best_arm] + best_arm_reward)/(operation[best_arm] + 1)
        # 更新摇臂操作次数
        operation[best_arm] += 1
        # 更新累积奖励
        total_reward += best_arm_reward
    return total_reward, expect_reward_estimate, operation

if __name__ == '__main__':
    N = 1000
    total_reward, expect_reward, operation_times = epsilon_greedy(N, explorate_rate)
    print("ε-greedy策略的累积奖励：", total_reward)
    # 期望奖励估计表
    expect_reward_table = pd.DataFrame({
        '期望奖励': expect_reward,
        '操作次数': operation_times
    })
    print(expect_reward_table)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 在0到1之间生成100个探索率
    explore_grad = np.arange(0.01, 1.01, 0.01)
    # 在不同探索率下，ε-greedy策略的累积奖励
    reward_result = [epsilon_greedy(N, i)[0] for i in explore_grad]
    # 绘制折线图
    plt.figure(figsize=(8, 6))
    plt.plot(explore_grad, reward_result, c='deepskyblue')
    plt.xlabel('探索率', fontsize=12)
    plt.ylabel('累积奖励', fontsize=12)
    plt.xlim(0, 1)
    plt.show()