#!/user/bin/env python3
# -*- conding:utf-8 -*-
# @auther Zhang
# @date 2021/8/12
# @file test.py
import pandas as pd
import numpy as np
n_state = 3
actions = ['a','b','c']
table = pd.DataFrame(
        np.zeros((n_state,len(actions))), # Q_table initial value
        columns=actions, # action's name
    )
print(table)