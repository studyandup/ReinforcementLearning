#!/usr/bin/env Ancconda
# -*- coding:utf-8 -*-
# @author Zhang
# @date 2021/8/2
# @file numpyTest.py

import numpy as np
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
df = DataFrame(np.arange(10).reshape(2,5))
print(df)
print('\n')
print(df.iloc[[0]])
print(df[0][1])
