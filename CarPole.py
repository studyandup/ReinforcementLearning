#!/user/bin/env python3
# -*- conding:utf-8 -*-
# @auther Zhang
# @date 2021/8/10
# @file CarPole.py
import gym

env = gym.make('CartPole-v0')
env.reset()
env.render()