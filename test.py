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
use_gpu = torch.cuda.is_available()
print(use_gpu)
