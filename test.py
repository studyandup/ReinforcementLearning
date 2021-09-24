#!/user/bin/env python3
# -*- conding:utf-8 -*-
# @auther Zhang
# @date 2021/8/12
# @file test.py
import pandas as pd
import numpy as np
import torch

# import matplotlib.pyplot as plt
# episode=[1,2,3]
# episode.append(4)
# epr=[4,5,6]
# epr.append(7)
#
# plt.plot(episode, epr)
# plt.show()
# use_gpu = torch.cuda.is_available()
# print(use_gpu)

# eps = np.finfo(np.float32).eps.item()  # 非负的最小值，使得归一化时分母不为0
# returns = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
# returns = torch.tensor(returns)
# returns = (returns - returns.mean()) / (returns.std() + eps)
# print(returns.mean(),returns.std())
# print(returns)

# x = torch.Tensor([1, 2, 3])
# print("x: ", x)
# x = x.unsqueeze(1)
# print("x.unsqueeze(1): \n", x)
# print("[ x*2 for i in range (1,4) ] : \n", [x * 2 for i in range(1, 4)])
# x2 = torch.cat([x * 2 for i in range(1, 4)], 1)
# print("x2 : \n", x2)
# x3 = torch.cat([x * 2 for i in range(1, 4)], 0)
# print("x3 : \n", x3)
# print(x3.size())
x = torch.tensor([-0.2015, -0.1247])
torch.unsqueeze(x,0)
print(x)


