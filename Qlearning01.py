#!/user/bin/env python3
# -*- conding:utf-8 -*-
# @auther Zhang
# @date 2021/8/12
# @file Qlearning01.py
# 来自 https://www.bilibili.com/video/BV13W411Y75P?p=6

import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATE = 6  # the length of the 1 dimensional world
ACTIONS = ['left', 'right']  # available action
EPSILON = 0.9  # greedy policy
ALPHA = 0.1  # learning rate
LAMBDA = 0.9  # discount factor
MAX_EPISODES = 13  # maximum episodes
FRESH_TIME = 0.3  # fresh time for one move


# initial Q_table
def build_Q_table(n_state, actions):
    table = pd.DataFrame(
        np.zeros((n_state, len(actions))),  # Q_table initial value
        columns=actions,  # action's name
    )
    print('Q_table:\n', table)  # show table
    return table


# if __name__ == '__main__':
#     df = build_Q_table(N_STATE,ACTIONS)
#     print(df.loc[0]['right'])

# this is choose how to action
def choose_action(state, q_table):
    state_action = q_table.loc[state, :]
    # 探索
    if (np.random.uniform() > EPSILON) or (state_action.all() == 0):
        action_name = np.random.choice(ACTIONS)
    # 利用
    else:
        action_name = state_action.idxmax()
    return action_name


# this is how agent will interact with the enviroment
def get_env_feedback(S, A):
    global S_
    if A == 'right':  # move right
        if S == N_STATE - 2:  # termminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:  # move left
        R = 0
        if S == 0:
            S_ = S
        else:
            S = S - 1  # reach the wall
    return S_, R


# this is how enviroment be updateed
def update_env(S, episode, step_counter):
    env_list = ['-'] * (N_STATE - 1) + ['T']  # '-------T' our enviroment
    if S == 'terminal':
        interaction = 'Episode %s :total_step = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                  ', end='')
    else:
        env_list[S] = 'o'
        interactions = ''.join(env_list)
        print('\r{}'.format(interactions), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_Q_table(N_STATE, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table[A][S]

            if S_ != 'terminal':
                q_target = R = LAMBDA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_
            update_env(S, episode, step_counter + 1)
            step_counter += 1
    return q_table


if __name__ == '__main__':
    q_table = rl()
    print('\r\n Q_table:\n')
    print(q_table)
