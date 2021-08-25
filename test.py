#!/user/bin/env python3
# -*- conding:utf-8 -*-
# @auther Zhang
# @date 2021/8/12
# @file test.py
import pandas as pd
import numpy as np
import torch

print(torch.__version__)
print("gpu", torch.cuda.is_available())