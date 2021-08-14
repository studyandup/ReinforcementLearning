#!/user/bin/env python3
# -*- conding:utf-8 -*-
# @auther Zhang
# @date 2021/8/6
# @file Boltzmann.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
ε-greedy策略并不是一个高效的算法，它基于“先训练再测试”的设定，只要摇臂被认为不是“当前最佳”的摇臂，在它们上面分配的次数就是一样多的。
然后我们尝试Boltzmann策略，它的核心思想是不应该根据摇臂能够带来多大的收益来对每一个摇臂分配实验次数，而应该根据它“有多大可能是最佳的摇臂”，
Boltzmann探索策略以[公式]归一化后的概率操作第 个摇臂，它是根据Boltzmann分布进行摇臂的选择：

'''
# 设定探索率
# 设摇臂个数为5，编号为0到4
k = 5
# 每个摇臂的真实奖励概率
real_reward = [0.2, 0.4, 0.7, 0.5, 0.3]
# 探索率设置为0.1
explorate_rate = 0.1
# sigma是Boltzmann的参数
sigma = 0.1


def boltzmann(N, sigma):
    # 初始化各摇臂期望奖励估计
    expect_reward_estimate = [0] * k
    # 初始化各摇臂操作次数
    operation = [0] * k
    # 初始化总奖励
    total_reward = 0
    for i in range(N):
        # np. exp(x)   e的x幂次方
        '''
         numpy.random.choice(a, size=None, replace=True, p=None)
         从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
         replace:True表示可以取相同数字，False表示不可以取相同数字
         数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
         '''
        # 通过Boltzmann分布计算摇臂的奖励概率
        reward_prob = np.exp(np.array(expect_reward_estimate) / sigma) / \
                      np.exp(np.array(expect_reward_estimate) / sigma).sum()
        # 通过奖励概率的分布进行抽样，选择摇臂
        best_arm = np.random.choice(k, size=1, p=reward_prob)[0]
        # 收集反馈的奖励数据
        best_arm_reward = np.random.binomial(1, real_reward[best_arm], size=1)[0]

        # 更新期望奖励估计
        expect_reward_estimate[best_arm] = (expect_reward_estimate[best_arm] * operation[
            best_arm] + best_arm_reward) / (operation[best_arm] + 1)
        # 更新摇臂操作次数
        operation[best_arm] += 1
        # 更新累积奖励
        total_reward += best_arm_reward
    return total_reward, expect_reward_estimate, operation


if __name__ == '__main__':
    N = 1000
    total_reward, expect_reward, operation_times = boltzmann(N, sigma)
    print("Boltzmann策略的累积奖励：", total_reward)
    # 期望奖励估计表
    expect_reward_table = pd.DataFrame({
        '期望奖励': expect_reward,
        '操作次数': operation_times
    })
    print(expect_reward_table)

    # 绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号“-”显示为方块的问题
    '''
    
    # 在0到1之间生成100个sigma
    sigma_grad = np.arange(0.01, 1.01, 0.01)
    # 得到在这100个sigma下，Boltzmann策略的累积奖励
    reward_result = [boltzmann(N, i)[0] for i in sigma_grad]
    # 绘制折线图
    plt.figure(figsize=(8, 6))
    plt.plot(sigma_grad, reward_result, c='firebrick')
    plt.xlabel('探索率', fontsize=12)
    plt.ylabel('累积奖励', fontsize=12)
    plt.xlim(0, 1)
    plt.show()
    '''
