# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 13:51:06 2021

@author: Haoyuan Yang

The mathematical modeling contest in 2020

Map NO.3

Reinforcement Learning in SARSA

"""

import random
import numpy as np
from No3_map import Map

class sarsa_agent(object):
    def __init__(self,Map):
        self.Map=Map
        self.Q={}
        self.E={}
        self.pro_fine=None
        self.state=0
        self.total_reward=0
        return
    def _init_agent(self):
        self.state=0
        return
    
    def get_weather(self,pro_fine):
        #随机获取天气，由输入的（气温晴朗）概率决定
        #0代表好天气 1代表高温
        if pro_fine is not None:
            randomnumnber=random.random()
            if randomnumnber <= pro_fine:
                return 0
            else:
                return 1
        else:
            print("天气生成失败")
            return None
    
    def get_action(self,state_0):
        #探索state_0的动作空间，也就是下一个状态
        action=[]
        for i in range(13):
            if Map[state_0,i]==1:
                action.append(i)
        return action
    
    def get_reward(self,state_0,state_1,weather):
        #是否需要对到达终点的行为进行奖励
        if state_0 >=13 or state_1 >= 13:
            print("超出地图")
            return None
        if weather==0:
            consume=-(2.5*3+5*4)
        else:
            consume=-(2.5*9+5*9)
        if state_0==state_1:
            #说明没有移动,需要判断能否挖矿
            if state_1==8:
                #能挖矿
                consume=consume*3
                reward=consume+200
                return reward
            else:
                #不能挖矿
                reward=consume
                return reward
        else:
            #说明移动了
            consume=consume*2
            reward=consume
            return reward
    
    def _init_Q(self):
        for i in range(2):
            self.Q[i]={}
            for j in range(13):
                self.Q[i][j]={}
                for k in self.get_action(j):
                    self.Q[i][j][k]=0
        return
    
    def _init_E(self):
        for i in range(2):
            self.E[i]={}
            for j in range(13):
                self.E[i][j]={}
                for k in self.get_action(j):
                    self.E[i][j][k]=0
        return
    
    def act(self,state_1,weather):
        is_done=False
        state_0=self.state
        self.state=state_1
        reward=self.get_reward(state_0, state_1, weather)
        if state_1==12:
            is_done=True
        return state_1,reward,is_done
    
    '''
    def act_random(self,weather):
        is_done=False
        action=self.get_action(self.state)
        ra=random.randint(0, len(action)-1)
        state_1=action[ra]
        state_0=self.state
        self.state=state_1
        if state_1==12:
            is_done=True
        reward=self.get_reward(state_0, state_1, weather)
        return state_1,reward,is_done
    '''
    
    def update_Q(self,theta,alpha):
        #根据参数更新Q
        for i in self.Q:
            for j in self.Q[i]:
                for k in self.Q[i][j]:
                    self.Q[i][j][k]=self.Q[i][j][k]+alpha*theta*self.E[i][j][k]
        return
    
    def update_E(self,gamma,lambda_):
        #根据参数更新E
        for i in self.E:
            for j in self.E[i]:
                for k in self.E[i][j]:
                    self.E[i][j][k]=gamma*lambda_*self.E[i][j][k]
    
    def cur_policy(self,s,num_episode,total_episode,weather,use_epsilon):
        epsilon=1-num_episode/total_episode
        if epsilon > 0.1:
            pass
        else:
            epsilon=0.1
        Q_s=self.Q[weather][s]
        rand_value=random.random()
        if use_epsilon and rand_value < epsilon:
            action=self.get_action(s)
            ra=random.randint(0, len(action)-1)
            return action[ra]
        else:
            return int(max(Q_s,key=Q_s.get))
        return
    
    def perform_policy(self,s,num_episode,total_episode,weather,use_epsilon=True):
        #主要问题是异策略怎么实现
        return self.cur_policy(s,num_episode,total_episode,weather,use_epsilon)
    
    def learning(self,pro_fine,lambda_,gamma,alpha,max_episode_num):
        num_episode=0
        self._init_Q()
        self._init_agent()
        while num_episode <= max_episode_num:
            num_episode+=1
            self._init_agent()
            self._init_E()
            #days_in_episode=1
            episodic_reward=0
            state_0=self.state
            weather_0=self.get_weather(pro_fine)
            state_1=self.perform_policy(self.state,num_episode,max_episode_num,weather_0,True)
            print("第一天：{}".format(state_0))
            for i in range(10):
                state_1,R,is_done=self.act(state_1, weather_0)
                #判断结束与否：
                print("第{}天：{}".format(i+1,state_1))
                print(is_done)
                if is_done:
                    episodic_reward+=R
                    break
                else:
                    if i == 9:
                        R=R-130 #作为没有完成任务的惩罚，！！！需要重新设计（或者是到达终点的奖励）
                    episodic_reward+=R
                    state_2=self.perform_policy(state_1, num_episode, max_episode_num, weather_0)
                    weather_1=self.get_weather(pro_fine)
                    theta=R+gamma*self.Q[weather_1][state_1][state_2]-self.Q[weather_0][state_0][state_1]
                    self.E[weather_0][state_0][state_1]=self.E[weather_0][state_0][state_1]+1
                    self.update_Q(theta,alpha)
                    self.update_E(gamma,lambda_)
                    #更新状态和天气
                    state_0=state_1
                    state_1=state_2
                    weather_0=weather_1
            print("训练次数：{}".format(num_episode))
            if is_done:
                print("到达终点")
            else:
                print("未到达终点")
            print("获得总奖励值：{}".format(episodic_reward))
        return self.Q


a=sarsa_agent(Map)
Q_test=a.learning(pro_fine=0.8,
                  lambda_=0.01,
                  gamma=0.8,
                  alpha=0.1,
                  max_episode_num=100000)
