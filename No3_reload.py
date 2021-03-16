# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 19:27:40 2021

@author: Haoyuan Yang

The mathematical modeling contest in 2020

Map NO.3

Reinforcement Learning in Q-learning （off-policy）

"""

import numpy as np
import random
from No3_map import Map

def get_action(pos_0):
        #探索pos_0的动作空间，也就是下一个状态
        action=[]
        for i in range(13):
            if Map[pos_0,i]==1:
                action.append(i)
        return action

#分别是：位置，天气，天数，动作
Q_value=np.load('FineWeather_80__Traning_200000.npy')
pro_fine=0.8 #因为上述数据是按照0.8训练的

pos_0=0

for i in range(10):
    
    rand_value=random.random()
    if rand_value<pro_fine:
        weather=0
    else:
        weather=1
    if pos_0==12:
        break
    Q_s=Q_value[pos_0][weather][i]
    action=get_action(pos_0)
    pos_1=action[0]
    value=Q_s[pos_1]
    for j in action:
        if Q_s[j] > value:
            value=Q_s[j]
            pos_1=j
        else:
            continue
    print("第{}天，从{}到{}，天气状况（0表示晴天，1表示炎热）：{}".format(i,pos_0,pos_1,weather))
    pos_0=pos_1