# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:12:02 2021

@author: Haoyuan Yang

毕业设计：穿越沙漠 第二题 第四关 格子世界环境
"""
#负向奖励版本
import gym
import random
import numpy as np
import pickle
import time

pro_SandStorm=0.1
epsilon_lower_bound=0.1

env=gym.make('GrandGridWorld-v0')
#env=env.DessertCrossing()
env=env.DessertCrossing_simple() #一个简化过后的环境，为了保证训练的效率
env.reset()

#在整个DessertCrossing环境中，动作空间是分离的
#0：左  1：右  2：上  3：下  4：不动
#state=x+n_width*y  坐标轴原点在左下角

class Sarsa_Agent(object):
    def __init__(self,):
        self.env=env          #环境设置为格子世界
        self.Q={}
        self.E={}
        self.pro_fine=None
        self.pro_SandStorm=pro_SandStorm
        self.water=240
        self.food=240
        #用一个很大的负值告诉他不能这样走
        self.reward_tabular=np.array([[-22.5,-67.5,-75],
                                      [-55,-135,-10000],
                                      [-82.5+1000,-202.5+1000,-225+1000]])
        return
    
    def _init_agent(self):
        self.env.reset()
        self.water=240
        self.food=240
        return
    
    def get_weather(self):
        #注意，这里有三种不一样的天气
        #小概率发生沙尘暴(2)
        #如何另外两种天气随机(0,1)
        if self.pro_fine is None:
            print("天气生成失败")
            return None
        rand_value_1=random.random()
        rand_value_2=random.random()
        if rand_value_1 <= self.pro_SandStorm:
            #生成沙尘暴
            return 2
        if rand_value_2 <= self.pro_fine:
            return 0
        else:
            return 1
        
    def _name_state(self,state):
        #将状态(数组)转化为字符串
        return str(state)
    
    def _assert_state(self,state_name,randomized=True):
        if not self._is_state_in_Q(state_name):
            self._init_state_value(state_name,randomized)
    
    def _is_state_in_Q(self,state_name):
        return self.Q.get(state_name) is not None
    
    def _init_state_value(self,state_name,randomized=True):
        if not self._is_state_in_Q(state_name):
            self.Q[state_name],self.E[state_name]={},{}
            for action in range(self.env.action_space.n):
                default_v=random.random()/10 if randomized is True else 0.0
                self.Q[state_name][action]=default_v
                self.E[state_name][action]=0.0
        return
    
    def E_copy_Q(self):
        #这个函数是为了reload 准备的
        #如果需要reload训练好的模型（Q表），则需要复制一下
        for state in self.Q:
            self.E[state]={}
            for action in range(self.env.action_space.n):
                self.E[state][action]=0.0
    
    def _get_(self,QorE,state_name,pos_1):
        self._assert_state(state_name,randomized=True)
        return QorE[state_name][pos_1]
    
    def _set_(self,QorE,state_name,pos_1,value):
        self._assert_state(state_name,randomized=True)
        QorE[state_name][pos_1]=value
    
    def _reset_E(self):
        #重新赋值为0，千万不要初始化
        for value_dic in self.E.values():
            for action in range(self.env.action_space.n):
                value_dic[action]=0.00
    
    
    def update_Q(self,theta,alpha):
        #根据参数更新Q
        for state in self.Q:
            for pos_1 in self.Q[state]:
                self.Q[state][pos_1]=self.Q[state][pos_1]+alpha*theta*self.E[state][pos_1]
    
    def update_E(self,gamma,lambda_):
        #根据参数更新E
        for state in self.E:
            for pos_1 in self.E[state]:
                self.E[state][pos_1]=gamma*lambda_*self.E[state][pos_1]
    
                
    def policy(self,state_name,epsilon):
        self._assert_state(state_name,randomized=True)
        Q_s=self.Q[state_name]
        rand_value=random.random()
        if rand_value <= epsilon:
            action=self.env.action_space.sample()
        else:
            action= int(max(Q_s,key=Q_s.get))
        return action
    
    def get_reward(self,observation_0,observation_1,weather_0):
        #除了返回reward，同时也要扣除相应的资源
        #先给出基础消耗
        if weather_0 == 0:
            water_consume=3
            food_consume=4
        elif weather_0 == 1:
            water_consume=9
            food_consume=9
        else:
            water_consume=10
            food_consume=10
        
        #村庄优先级最高，防止出现多个返回值
        if observation_1 == 13:
            #按照行走三天高温购买一个基础量
            self.water += 54
            self.food  += 54
            return -1620
        
        #如果停留
        if observation_0 == observation_1:
            #考虑挖矿与否
            if observation_0 == 7:
                self.water -= water_consume*3
                self.food -= food_consume*3
                return self.reward_tabular[2,weather_0]
            else:
                self.water -= water_consume
                self.food -= food_consume
                return self.reward_tabular[0,weather_0]
        else:
            #即行动
            self.water -= water_consume*2
            self.food -= food_consume*2
            return self.reward_tabular[1,weather_0]
        
        
        
    
    def learning(self,lambda_,gamma,alpha,max_episode_num):
        #开始学习
        #如果不初始化，是否可以继续学习？
        self._init_agent()
        time_start=time.time()
        for num_episode in range(max_episode_num):
            #这个部分可以展示训练进度
            present_percent_0=int((num_episode*100/max_episode_num)*100)/100
            present_percent_1=int(((num_episode+1)*100/max_episode_num)*100)/100
            if present_percent_0!=present_percent_1:
                print("训练进度：{:.2f}%".format(present_percent_0))
            #首先创建一个线性下降的epsilon
            epsilon=max(1-num_episode/max_episode_num,epsilon_lower_bound)
            
            self._init_agent()
            self._reset_E()
            total_reward=0
            self.pro_fine=random.random()
            weather_0=self.get_weather()
            observation_0=self.env.state
            water=int(self.water/240*10)
            food=int(self.food/240*10)
            state_0=self._name_state([observation_0,weather_0,0,water,food])
            
            present=False
            if num_episode/max_episode_num >0.9999:
                present=True
            
            if present:
                self.env.render()
                
            for i in range(30):
                action_0=self.policy(state_0, epsilon)
                observation_1,reward,is_done,info=env.step(action_0)
                if present:
                    self.env.render()
                    print("第{}天，到达{}，天气{}".format(i,observation_1,weather_0))
                #两件事：修改reward和自身状态
                reward=self.get_reward(observation_0, observation_1, weather_0)
                
                if is_done and present:
                    print("到达终点")
                
                if self.water <=0 or self.food <= 0:
                    is_done=True
                    if present:
                        print("未到达终点")
                    reward -= 10000 #未到达终点的惩罚
                if i == 29 and observation_1!= 4:
                    is_done=True
                    if present:
                        print("未到达终点")
                    reward -= 10000 #未到达终点的惩罚
                
                total_reward += reward
                weather_1=self.get_weather()
                water=int(self.water/240*10)
                food=int(self.food/240*10)
                state_1=self._name_state([observation_1,weather_1,i+1,water,food])
                action_1=self.policy(state_1, 0) #这里的更新是greedy的
                old_q=self._get_(self.Q, state_0, action_0)
                new_q=self._get_(self.Q, state_1, action_1)
                if is_done:
                    theta=reward
                else:
                    theta=reward+gamma*new_q-old_q
                E_value=self._get_(self.E, state_0, action_0)+1
                self._set_(self.E, state_0, action_0, E_value)
                self.update_Q(theta, alpha)
                self.update_E(gamma, lambda_)
                observation_0=observation_1
                weather_0=weather_1
                state_0=state_1
                if is_done:
                    if present:
                        print("总奖励值：{}".format(total_reward))
                    break
            time_end=time.time()
        print("训练总时间{}".format(time_end-time_start))
        return self.Q

a=Sarsa_Agent()
Q=a.learning(lambda_=0.1, 
             gamma=0.8, 
             alpha=0.001, 
             max_episode_num=100000)

file = open('NO4_10w_negetive_alpha_10-3.pickle', 'wb')
pickle.dump(Q, file)
file.close()
