# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:05:06 2021

@author: 96174
"""

#基于Q表值的测试与展示

import ChasingAndBlocking as CAB
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import gym
import random

env=gym.make('MA-GrandGridWorld-v0')
env=env.ChasingAndBlocking()

a=CAB.chess_agent()
file_name="ChasingAndBlocking_Q-learning_500w_model_30_step.pickle"
with open(file_name, 'rb') as file:
    a.Q=pickle.load(file)
    
test_times=10

for i in range(test_times):
    print("第{}次评估开始".format(i))
    a._init_agent()
    print(a.pos_all)
    env.set_pos(a.pos_all)
    time.sleep(1)
    is_done=False
    iteration=0
    while is_done == False and iteration <= 100:
        env.render()
        action_thief=random.randint(0, 4)
        state=a.get_state(action_thief)
        a.thief_act(action_thief) #已经收集动作，小偷可以行动
        union_action=a.policy(state,0)
        pos_all,reward,is_done=a.union_act(union_action)
        env.set_pos(a.pos_all)
        env.render()
        time.sleep(0.2)
        print("第{}步".format(iteration))
        if is_done:
            print("第{}次评估成功".format(i))
            time.sleep(1)
            #env.close()
            break
        iteration+=1
env.close()
        