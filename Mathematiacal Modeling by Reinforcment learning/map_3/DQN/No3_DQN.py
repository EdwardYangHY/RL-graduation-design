# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 18:12:07 2021

@author: 96174
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from No3_map import Map

BATCH_SIZE=32
LR=0.01
EPSILON=0.9
GAMMA=0.9
TARGET_REPLACE_ITER=100        #目标更新频率/步数
MEMORY_CAPACITY=1000

input_len=5
#输入数据（state，a,r,s',a'）输出一个值

class Net(nn.Module):
    def __init__(self, ):
        super(Net,self).__init__()
        self.fc1=nn.Linear(input_len+1, 50)
        self.fc1.weight.data.normal_(0,0.1) #参数初始化
        self.out=nn.Linear(50,1)
        self.out.weight.data.normal_(0,0.1) #参数初始化
        
    def forward(self,x):
        #输入x是一个张量，构成是[state,a]
        x=self.fc1(x)
        x=F.relu(x)
        action_value=self.out(x)
        return action_value

class DQN(object):
    def __init__(self):
        self.Map=Map
        self.pos=0
        self.pro_fine=None
        self.state=None
        self.total_reward=0
        self.water=240
        self.food=240
        
        self.eval_net, self.target_net = Net(),Net()
        self.learn_step_counter=0         #目标更新，学习步数
        self.memory_counter=0             #存储更新
        self.memory=np.zeros((MEMORY_CAPACITY,input_len*2+2))
        self.optimizer=torch.optim.Adam(self.eval_net.parameters(),lr=LR)
        self.loss_func=nn.MSELoss()
    
    def _init_agent(self):
        self.pos=0
        self.state=None
        self.water=240
        self.food=240
        return
    
    def _get_action(self,pos_0): #
        #探索pos_0的动作空间，也就是下一个状态
        action=[]
        for i in range(13):
            if Map[pos_0,i]==1:
                action.append(i)
        if pos_0==8:
            #说明到达矿山，此时给出动作挖矿
            action.append(-1)
        return action
    
    def get_action(self,pos_0):
        #探索pos_0的动作空间，也就是下一个状态
        action=[]
        for i in range(13):
            if Map[pos_0,i]==1:
                action.append(i)
        return action
    
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
    
    def _get_reward(self,pos_0,pos_1,weather): #
        #是否需要对到达终点的行为进行奖励
        if pos_0 >=13 or pos_1 >= 13:
            print("超出地图")
            return None
        if weather==0:
            consume=-(2.5*3+5*4)
        else:
            consume=-(2.5*9+5*9)
        
        #####
        if pos_0==pos_1 or pos_1==-1:
            #说明没有移动,需要判断能否挖矿
            #改进以后（8，8）变成了停留，（8，-1）变成了挖矿
            if pos_1==-1:
                #能挖矿
                consume=consume*3
                reward=consume+200
                return reward
            else:
                #不能挖矿
                reward=consume
                return reward
         #####   
            
        else:
            #说明移动了
            consume=consume*2
            reward=consume
            return reward
        
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
        
    def act(self,pos_1,weather):
        #此时act是无法改变自身状态（self.state）的，因为状态需要天气和天数
        is_done=False
        pos_0=self.pos
        #如果挖矿那么自身状态不改变
        if pos_1!=-1:
            self.pos=pos_1
        
        reward=self.get_reward(pos_0, pos_1, weather)
        if reward==-55.0:
            self.water-=6
            self.food-=8
        if reward==-135.0:
            self.water-=18
            self.food-=18
        if reward==-27.5:
            self.water-=3
            self.food-=4
        if reward==-67.5:
            self.water-=9
            self.food-=9
        if reward==117.5:
            self.water-=9
            self.food-=12
        if reward==-2.5:
            self.water-=27
            self.food-=27
        if pos_1==12:
            is_done=True
        return pos_1,reward,is_done
    
    def store_transition(self,s,a,r,s_):
        #hstack是横向堆栈
        transiton=np.hstack((s,[a,r],s_))
        #超出上限以后新内存覆盖老内存
        index=self.memory_counter%MEMORY_CAPACITY
        self.memory[index,:]=transiton
        self.memory_counter+=1
    
    def choose_random(self,pos_0):
        action=self.get_action(pos_0)
        ra=random.randint(0, len(action)-1)
        out_action=action[ra]
        #print(ra)
        return out_action
    
    def choose_max(self,state):
        pos_0=int(state[0])
        action=self.get_action(pos_0)
        action_value=torch.zeros(len(action))
        for i in range(len(action)):
            x=np.append(state,[action[i]])
            x=torch.Tensor(x)
            action_value[i]=self.eval_net.forward(x)
        out_action=action[action_value.argmax()]
        #print(action_value)
        return out_action
    
    
    def choose_action_epsilon(self,state,epsilon):
        #实现可以控制的choose_action_epsilon
        #epsilon=0,则选择最大值，epsilon=1则选择随机值
        rand_value=random.random()
        if rand_value < epsilon:
            pos_0=int(state[0])
            return self.choose_random(pos_0)
        else:
            return self.choose_max(state)
        
        
    def learning(self):
        #以TARGET_REPLACE_ITER为频率更新target网络
        if self.learn_step_counter % TARGET_REPLACE_ITER:
            self.target_net.load_state_dict(self.eval_net.state_dict()) #eval参数复制到taget里
        self.learn_step_counter += 1
        
        sample_index=np.random.choice(MEMORY_CAPACITY,BATCH_SIZE)
        b_memory=self.memory[sample_index,:]                            #加载记忆
        #需要注意的是，下面的数据都是一个batch[32,]的数据
        
        b_sa=torch.FloatTensor(b_memory[:, :input_len+1])              #构架成功
        b_r = torch.FloatTensor(b_memory[:, input_len+1:input_len+2])
        
        #合并
        action_0=self.choose_max(b_memory[0, -input_len:])
        action_1=self.choose_max(b_memory[1, -input_len:])
        x_0=np.append(b_memory[0, -input_len:],action_0)
        x_1=np.append(b_memory[1, -input_len:],action_1)
        b_s_a=np.stack((x_0,x_1))
        for i in range(2,BATCH_SIZE):
            action=self.choose_max(b_memory[i, -input_len:])        #这里不应该是choose_action,应该是最大的动作值
            x=np.append(b_memory[i, -input_len:],action)
            b_s_a=np.insert(b_s_a,-1,x,0)
        #转化为tensor
        b_s_a=torch.FloatTensor(b_s_a)
        #print(b_s_a)
        
        #评估网络实时更新
        #目标网络每固定步长更新一次
        
        q_eval=self.eval_net(b_sa)
        q_next=self.target_net(b_s_a).detach()
        q_target=b_r+GAMMA*q_next
        #print(q_target)
        
        loss=self.loss_func(q_eval,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    
    def test(self,pro_fine,lambda_,gamma,alpha,max_episode_num):
        self._init_agent()
        self.pro_fine=pro_fine

        for num_episode in range(max_episode_num):
            self._init_agent()
            total_reward=0
            #不再对天气的概率做约束，是否就是我每次随机生成一组天气然后训练呢
            #新代码
            self.pro_fine=random.random() #这样每次生成都是随机的
            #新代码
            water=self.water
            food=self.food
            
            weather_0=self.get_weather()
            state_0=[self.pos,weather_0,0,water,food]
            pos_0=self.pos
            
            for i in range(10):
                #print("剩余水{}，剩余食物{}".format(water,food))
                pos_1=self.choose_random(pos_0)
                pos_1,reward,is_done=self.act(pos_1, weather_0)
                
                water=self.water
                food=self.food
                
                #还要判断是否花光了补给！
                if self.water<=0 or self.food<=0:
                    reward=reward-1000
                    #注意：想要展现出效果，需要将这个-1000的反馈值传递给Q表和E表！
                    theta=reward  #到达终点后没有下一个状态了，所以直接用reward传值给theta
                    '''
                    更新网络值？
                    '''
                    
                    
                    total_reward+=reward
                
                if i==9 and is_done==False:
                    reward=reward-1000
                    #注意：想要展现出效果，需要将这个-1000的反馈值传递给Q表和E表！
                    theta=reward  #到达终点后没有下一个状态了，所以直接用reward传值给theta
                    '''
                    更新网络值？
                    '''
                    
                    total_reward+=reward
                    break
                
                total_reward+=reward
                if is_done:
                    break

                weather_1=self.get_weather()
                #这个状态列表里不能有-1
                if pos_1==-1:
                    state_1=[8,weather_1,i+1,water,food]   #下一个时刻的状态S'
                else:
                    state_1=[pos_1,weather_1,i+1,water,food]   #下一个时刻的状态S'
                
                self.store_transition(state_0, pos_1, reward, state_1)
                pos_2=self.choose_random(pos_1)
                pos_0=pos_1
                pos_1=pos_2
                weather_0=weather_1  #更新天气
                state_0=state_1      #更新状态S
                if num_episode==0:
                    print(pos_1)
        #print(self.memory)
        return self.memory
                
    def training(self,lambda_,gamma,alpha,max_episode_num):
        self._init_agent()
        self.pro_fine=random.random()
        for num_episode in range(max_episode_num):
            #首先做一个逐步下降但是存在下界的epsilon
            epsilon=1-num_episode/max_episode_num
            #print(epsilon)
            epsilon_lowerbound=0.1
            if epsilon <= epsilon_lowerbound:
                epsilon=epsilon_lowerbound
            
            print("训练进度：{:.2f}%".format(num_episode*100/max_episode_num))
            
            self._init_agent()
            total_reward=0
            
            self.pro_fine=random.random()
            
            weather_0=self.get_weather()
            state_0=[self.pos,weather_0,0,self.water,self.food]
            
            for i in range(10):
                pos_0=self.pos #纯展示用
                pos_1=self.choose_action_epsilon(state_0, epsilon)
                if num_episode>=max_episode_num*0.99:
                    print("第{}天，天气{},从{}到{}".format(i,weather_0,pos_0,pos_1))
                pos_1,reward,is_done=self.act(pos_1, weather_0)
                '''
                if pos_0==pos_1 and pos_1==8:
                    reward=200
                elif pos_1==12:
                    reward=self.water*2.5+self.food*5
                else:
                    reward=0
                '''
                if self.water <=0 or self.food <= 0:
                    reward=reward-1000
                    is_done=True
                if i==9 and is_done==False:
                    reward=reward-1000
                    is_done=True
                total_reward+=reward
                
                weather_1=self.get_weather()
                if pos_1==-1:
                    state_1=[8,weather_1,i+1,self.water,self.food]
                else:
                    state_1=[pos_1,weather_1,i+1,self.water,self.food]
                #存储记忆
                self.store_transition(state_0, pos_1, reward, state_1)
                state_0=state_1
                weather_0=weather_1
                
                if is_done:
                    break
            
            if self.memory_counter >= MEMORY_CAPACITY:
                self.learning()
                
                

a=DQN()
'''
Q=a.test(pro_fine=0.6,
           lambda_=0.01,
           gamma=0.8,
           alpha=0.1,
           max_episode_num=4000)
'''
#Q_=a.learning()

Q=a.training(lambda_=0.01,
           gamma=0.8,
           alpha=0.1,
           max_episode_num=1000)

'''
x=[0,0,0,1,1,3]
x=torch.Tensor(x)
hh=a.eval_net.forward(x)

x=np.array([10,0,0,240,240])
b=a.choose_action(x)

'''
