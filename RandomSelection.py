'''
https://zhuanlan.zhihu.com/p/112970426
随机选择策略是解决强化学习较为简单的策略，它的核心思想是对每个摇臂操作一定的次数，
然后收集每个摇臂奖励的期望，基于此选择期望最大的摇臂作为最佳摇臂，
每次游戏都选择最佳摇臂进行操作，接下来实现随机选择策略，并进行1000次游戏。
'''
import numpy as np
import pandas as pd
# 设摇臂个数为5，编号为0到4
k = 5
# 每个摇臂的真实奖励概率
real_reward = [0.2, 0.4, 0.7, 0.5, 0.3]


def random_select(N):  # N为游戏次数
    # 初始化各摇臂期望奖励估计
    expect_reward_estimate = [0] * k
    # 初始化各摇臂操作次数
    operation = [0] * k
    # 初始化总奖励
    total_reward = 0

    for i in range(N):
        '''
        numpy.random.choice(a, size=None, replace=True, p=None)
        从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
        replace:True表示可以取相同数字，False表示不可以取相同数字
        数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
        '''

        # 随机选择一个摇臂进行操作
        arm = np.random.choice(k, size=1)[0]
        '''
        np.random.binomial(n, p, size=None)
        参数的介绍：
        n：一次试验的样本数n，并且相互不干扰。 有些博客解释这里的参数n是试验次数，
            我认为不对，或许现在看起来好像一次试验n个互不相干的样本和n次试验一个样本是一样的，
            但是如果把n理解为试验次数，那和size参数的意义会产生冲突；
            
        n表示n次的试验，p表示的试验成功的概率，n可是是一个float但是也会被变成整数来使用。
        
        p：事件发生的概率p，范围[0,1]。这里有个理解的关键就是 “事件发生”到底是指的什么事件发生？
            准确来讲是指：如果一个样本发生的结果要么是A要么是B，事件发生指的是该样本其中一种结果发生。
        size：限定了返回值的形式（具体见上面return的解释）和实验次数。当size是整数N时，表示实验N次，返回每次实验中事件发生的次数；
            size是（X，Y）时，表示实验X*Y次，以X行Y列的形式输出每次试验中事件发生的次数。
            （如果将n解释为试验次数而不是样本数的话，这里返回数组中数值的意义将很难解释）。
        return返回值 ： 以size给定的形式，返回每次试验事件发生的次数，次数大于等于0且小于等于参数n。
            注意：每次返回的结果具有随机性，因为二项式分布本身就是随机试验。
        
        该函数参照 https://blog.csdn.net/u014571489/article/details/102942933
        https://zhuanlan.zhihu.com/p/85663313 这个也有点意思
        '''
        # 收集反馈的奖励数据
        arm_reward = np.random.binomial(1, real_reward[arm], size=1)[0]
        print("收集反馈的奖励数据:", arm_reward)
        #
        # 更新期望奖励估计
        expect_reward_estimate[arm] = (expect_reward_estimate[arm] * operation[arm] + arm_reward) / (operation[arm] + 1)
        # 更新摇臂操作次数
        operation[arm] += 1
        # 更新累积奖励
        total_reward += arm_reward

    return total_reward, expect_reward_estimate, operation

if __name__ == '__main__':
    N = 10
    total_reward, expect_reward, operation_times = random_select(N)
    print("随机选择的累积奖励：", total_reward)
    # 期望奖励估计表
    expect_reward_table = pd.DataFrame({
        '期望奖励': expect_reward,
        '操作次数': operation_times
    })
    print(expect_reward_table)