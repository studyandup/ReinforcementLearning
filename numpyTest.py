#!/usr/bin/env Ancconda
# -*- coding:utf-8 -*-
# @author Zhang
# @date 2021/8/2
# @file numpyTest.py

import numpy as np
import torch
from pandas import DataFrame

'''
    轴 axis ：保存数据的维度
    秩 rank ： 轴的数量
    
    使用NumPy中函数来创建ndarray数组如：arange，ones,zeros等
'''
# a=np.array([[1,2,3],[4,5,6]])
# print(a[0])
# print(a.size)
# print(a.shape)
# print(a.dtype)
# print(a.itemsize)
# a = np.array(10)
# print(a)
# a=[1,2,3,4]
# print(a.index(3))
# print(np.random.uniform(0,1,(1,2)))
# df = DataFrame(np.arange(10).reshape(2,5))
# print(df)
# print('\n')
# print(df.iloc[[0]])
# print(df[0][1])
# -------------------------
# memory = np.ones((10, 10))
# index = np.random.choice(10, 3)
# test = memory[index, :]
# # print(memory)
# print(index)
# print(test)
# print(test[0])
# -----------------------
# t = torch.Tensor([[1,2],[3,4]])
# m = t.gather(1, torch.LongTensor([[0,0],[1,0]]))
# print(t)
# print(m)

q_next = torch.tensor([[0.1454, -0.1610],
                       [ 0.1349, -0.1114],
                       [ 0.1475, -0.1558],
                       [ 0.1443, -0.1535],
                       [ 0.1424, -0.1404],
                       [ 0.1415, -0.1279],
                       [ 0.1446, -0.1467],
                       [ 0.1404, -0.1354],
                       [ 0.1429, -0.1422],
                       [ 0.1420, -0.1352],
                       [ 0.1458, -0.1543],
                       [ 0.1376, -0.1239],
                       [ 0.1453, -0.1635],
                       [ 0.1371, -0.1313],
                       [ 0.1451, -0.1570],
                       [ 0.1447, -0.1592],
                       [ 0.1389, -0.1326],
                       [ 0.1377, -0.1242],
                       [ 0.1427, -0.1461],
                       [ 0.1409, -0.1428],
                       [ 0.1446, -0.1545],
                       [ 0.1424, -0.1440],
                       [ 0.1403, -0.1353],
                       [ 0.1384, -0.1267],
                       [ 0.1456, -0.1645],
                       [ 0.1426, -0.1571],
                       [ 0.1442, -0.1441],
                       [ 0.1465, -0.1602],
                       [ 0.1452, -0.1644],
                       [ 0.1455, -0.1557],
                       [ 0.1425, -0.1454],
                       [ 0.1449, -0.1657]])
print(q_next.max(1)[0].view(32, 1))