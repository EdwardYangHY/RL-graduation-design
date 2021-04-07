# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 19:27:40 2021

@author: Haoyuan Yang

The mathematical modeling contest in 2020

Map NO.3

Reinforcement Learning in Q-learning （off-policy）

"""
#本文件在天气固定概率下训练agent


import random
import numpy as np
from No3_map import Map

class sarsa_agent(object):
    def __init__(self,Map):
        self.Map=Map
        self.Q={}
        self.E={}
        self.pro_fine=None
        self.pos=0
        self.state=None
        self.total_reward=0
        return
    
    def _init_agent(self):
        self.pos=0
        self.state=None
        return
    
    def get_weather(self):
        #随机获取天气，由输入的（气温晴朗）概率决定
        #0代表好天气 1代表高温
        pro_fine=self.pro_fine
        if pro_fine is not None:
            randomnumnber=random.random()
            if randomnumnber <= pro_fine:
                return 0
            else:
                return 1
        else:
            print("天气生成失败")
            return None
    
    def get_action(self,pos_0):
        #探索pos_0的动作空间，也就是下一个状态
        action=[]
        for i in range(13):
            if Map[pos_0,i]==1:
                action.append(i)
        return action
    
    def get_reward(self,pos_0,pos_1,weather):
        #是否需要对到达终点的行为进行奖励
        if pos_0 >=13 or pos_1 >= 13:
            print("超出地图")
            return None
        if weather==0:
            consume=-(2.5*3+5*4)
        else:
            consume=-(2.5*9+5*9)
        if pos_0==pos_1:
            #说明没有移动,需要判断能否挖矿
            if pos_1==8:
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
            state_list=eval(state_name)
            action_space=self.get_action(state_list[0]) #获取状态空间
            for pos_1 in action_space:
                default_value=random.random() if randomized is True else 0.0
                self.Q[state_name][pos_1]=default_value
                self.E[state_name][pos_1]=0.0
        return
    
    def _get_(self,QorE,state_name,pos_1):
        self._assert_state(state_name,randomized=True)
        return QorE[state_name][pos_1]
    
    def _set_(self,QorE,state_name,pos_1,value):
        self._assert_state(state_name,randomized=True)
        QorE[state_name][pos_1]=value
    
    def _reset_E(self):
        #重新赋值为0，千万不要初始化
        if self.E is None:
            pass
        for state in self.E:
            for pos_1 in self.E[state]:
                self.E[state][pos_1]=0.0


    def act(self,pos_1,weather):
        #此时act是无法改变自身状态（self.state）的，因为状态需要天气和天数
        is_done=False
        pos_0=self.pos
        self.pos=pos_1
        reward=self.get_reward(pos_0, pos_1, weather)
        if pos_1==12:
            is_done=True
        return pos_1,reward,is_done
    
    
    
    #policy也需要做相应的修改
    def policy(self,state_name,num_episode,total_episode,use_epsilon):
        #首先编写epsilon-greedy的算法
        #这里沿用deepmind的思路，做一个渐进下降到0.1（0.01也可以）的epsilon
        lower_bound=0.1 #设置epsilon的下限以保证探索性
        epsilon=1-num_episode/total_episode
        if epsilon > lower_bound:
            pass
        else:
            epsilon=lower_bound
        self._assert_state(state_name,randomized=True)
        Q_s=self.Q[state_name]
        rand_value=random.random()
        if use_epsilon and rand_value<epsilon:
            #随机生成一个动作
            pos_0=eval(state_name)[0] #获取当前位置
            action=self.get_action(pos_0) #获取动作空间
            ra=random.randint(0, len(action)-1)
            return action[ra]
        else:
            return int(max(Q_s,key=Q_s.get))
        
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
                    
    def test(self):
        #先测试一下随机游动是否能正确地将状态压入表值中
        is_done=False
        self.pro_fine=0.8
        self._init_agent()
        for i in range(10):
            weather_0=self.get_weather()
            state_0=[self.pos,weather_0,i]
            state_name_0=str(state_0)
            pos_1=self.policy(state_name_0,0,10000,True)
            pos_0=self.pos
            pos_1,reward,is_done=self.act(pos_1, weather_0)
            print("第{}天，天气{}，从{}走到{}，获得奖励：{}".format(i,weather_0,pos_0,pos_1,reward))
            
    def learning(self,pro_fine,lambda_,gamma,alpha,max_episode_num):
        self.Q={}
        self.E={}
        self._init_agent()
        self.pro_fine=pro_fine
        for num_episode in range(max_episode_num):
            print("训练进度：{:.2f}%".format(num_episode*100/max_episode_num))
            self._init_agent()
            self._reset_E()
            total_reward=0
            weather_0=self.get_weather()
            state_0=[self.pos,weather_0,0]
            #pos_0=self.pos
            
            present=False
            if num_episode/max_episode_num >0.9:
                present=True
            
            for i in range(10):
                pos_1=self.policy(str(state_0), num_episode, max_episode_num, use_epsilon=True) #这里是行动策略，epsilon-greedy
                if present:
                    print("第{}天，从{}到{}，天气{}".format(i,self.pos,pos_1,weather_0))
                pos_1,reward,is_done=self.act(pos_1, weather_0)
                
                
                if i==9 and is_done==False:
                    reward=reward-1000
                    #注意：想要展现出效果，需要将这个-1000的反馈值传递给Q表和E表！
                    theta=reward  #到达终点后没有下一个状态了，所以直接用reward传值给theta
                    E_value=self._get_(self.E, str(state_0), pos_1)
                    self._set_(self.E, str(state_0), pos_1, E_value+1)
                    self.update_Q(theta, alpha)
                    self.update_E(gamma, lambda_)
                    total_reward+=reward
                    if present:
                        print("未到达终点，总奖励：{:.2f}".format(total_reward))
                    break
                
                total_reward+=reward
                if is_done:
                    if present:
                        print("到达终点，总奖励：{:.2f}".format(total_reward))
                    break

                weather_1=self.get_weather()
                state_1=[pos_1,weather_1,i+1]   #下一个时刻的状态S'
                pos_2=self.policy(str(state_1), num_episode, max_episode_num, use_epsilon=False) #目标更新，greedy
                theta=reward+gamma*self._get_(self.Q, str(state_1), pos_2)-self._get_(self.Q, str(state_0), pos_1)
                E_value=self._get_(self.E, str(state_0), pos_1)
                self._set_(self.E, str(state_0), pos_1, E_value+1)
                self.update_Q(theta, alpha)
                self.update_E(gamma, lambda_)
                #pos_0=pos_1
                #pos_1=pos_2          #更新动作A
                weather_0=weather_1  #更新天气
                state_0=state_1      #更新状态S
        return self.Q
            

def convert_Q(Q):
    #将获得的值存储在一个numpy数组中，以便于存储到文件
    #分别是：位置，天气，天数，动作
    Q_np=np.zeros((13,2,10,13))
    for state in Q:
        state_np=eval(state)
        for pos_1 in Q[state]:
            Q_np[state_np[0],state_np[1],state_np[2],pos_1]=Q[state][pos_1]
    return Q_np

a=sarsa_agent(Map)
'''
Q=a.learning(pro_fine=0.6,
           lambda_=0.01,
           gamma=0.8,
           alpha=0.1,
           max_episode_num=200000)

#Q_np=convert_Q(Q)
#np.save('FineWeather_60__Traning_200000.npy',Q_np)
'''
for i in range(1,10):
    
    pro_fine=i/10
    max_episode_num=200000
    Q=a.learning(pro_fine,
                 lambda_=0.01,
                 gamma=0.8,
                 alpha=0.1,
                 max_episode_num=200000)
    Q_np=convert_Q(Q)
    pro_fine=round(pro_fine,2)
    np.save('Fineweather_'+str(pro_fine)+'__Training_'+str(max_episode_num)+'.npy',Q_np)
