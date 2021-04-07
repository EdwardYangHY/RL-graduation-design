# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:26:50 2020

@author: 96174
"""

import gym
import time

env=gym.make('GrandGridWorld-v0')
env=env.DessertCrossing_simple()
env.reset()
count=0
rewards=0
for i in range(100):
    action=env.action_space.sample()  #随机采样动作
    observation,reward,done,info=env.step(action) #与环境交互，获得下一步的时刻
    rewards+=reward
    print("目前位置{}，观测值{}，单步奖励：{}，选择动作：{}".format(env.state,observation,reward,action))
    if done:
        break
    env.render() #表示gym也开始更新
    count+=1
    time.sleep(0.1)
print(count)