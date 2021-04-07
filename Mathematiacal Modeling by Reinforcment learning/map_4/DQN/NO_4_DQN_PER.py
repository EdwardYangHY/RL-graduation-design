# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:12:02 2021

@author: Haoyuan Yang

毕业设计：穿越沙漠 第二题 第四关 格子世界环境

DQN+PER版本

输入仍然沿用第三关的设定
"""
#正向奖励版本
import gym
import random
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

BATCH_SIZE=32
LR=0.00001
#EPSILON=0.9
GAMMA=0.9
TARGET_REPLACE_ITER=100        #目标更新频率/步数
MEMORY_CAPACITY=2000

input_len=5

pro_SandStorm=0.1
epsilon_lower_bound=0.1

env=gym.make('GrandGridWorld-v0')
env=env.DessertCrossing()
#env=env.DessertCrossing_simple() #一个简化过后的环境，为了保证训练的效率
env.reset()

#在整个DessertCrossing环境中，动作空间是分离的
#0：左  1：右  2：上  3：下  4：不动
#state=x+n_width*y  坐标轴原点在左下角

class Net(nn.Module):
    def __init__(self, ):
        super(Net,self).__init__()
        self.fc1=nn.Linear(input_len, 50)
        self.fc1.weight.data.normal_(0,0.1) #参数初始化
        self.fc2=nn.Linear(50, 30)
        self.fc2.weight.data.normal_(0,0.1) #参数初始化
        self.out=nn.Linear(30,env.action_space.n)  #这里输出是五个动作
        self.out.weight.data.normal_(0,0.1) #参数初始化
        #self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        #输入x是一个张量，构成是[state,a]
        #尝试对x进行归一化处理
        #【pos,weather,day,water,food,action】
        #尽量避免0的出现
        if len(x.shape)==1:
            x[0]=(x[0]+1)/24
            x[1]=(x[1]+1)/3
            x[2]=(x[2]+1)/30
            x[3]=x[3]/240
            x[4]=x[4]/240
        else:
            x[:,0]=(x[:,0]+1)/24
            x[:,1]=(x[:,1]+1)/3
            x[:,2]=(x[:,2]+1)/30
            x[:,3]=x[:,3]/240
            x[:,4]=x[:,4]/240
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.relu(x)
        action_value=self.out(x)
        return action_value

class SumTree(object):
    #构建优先经验回放中的组件 Sumtree
    #该数据结构能够极大地减少运算量
    data_pointer=0 #一个指针
    
    def __init__(self,capacity):
        self.capacity=capacity
        self.tree=np.zeros(capacity+capacity-1) #二叉树中，父节点和叶子的关系
                                                #SumTree中存储每个结点的值（子结点的和）
        self.data=np.zeros(capacity,dtype=object)
    
    def add(self,priority,data):
        tree_index=self.data_pointer+self.capacity-1 #因为前self.capacity-1个index都是父结点
        self.data[self.data_pointer]=data
        self.update(tree_index,priority) #更新树结构的权重，为抽样做准备
        
        self.data_pointer+=1
        if self.data_pointer>=self.capacity:
            self.data_pointer=0
    
    def update(self,tree_index,priority):
        change=priority-self.tree[tree_index]
        self.tree[tree_index]=priority
        #如何通过指针将修改后的优先级回调，因为用了change，所以只需要传上change即可
        while tree_index!=0:
            tree_index=(tree_index-1)//2 #整除
            self.tree[tree_index]+=change
    
    def get_leaf(self,v):
        #输入抽样的得到的数据v，检索抽样的记忆编号
        parent_index=0
        while True:
            left_child_index=2*parent_index+1
            right_child_index=2*parent_index+2
            if left_child_index >= len(self.tree):
                #此时已经到子结点
                leaf_index=parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index=left_child_index
                else:
                    v-=self.tree[left_child_index]
                    parent_index=right_child_index
        data_index=leaf_index-(self.capacity-1)
        #返回该记忆在树中的index，priority的值和记忆片段
        return leaf_index,self.tree[leaf_index],self.data[data_index]
    
    @property
    def total_p(self):
        #用property装饰的，返回优先级值的和
        return self.tree[0]

class memory(object):
    #这里的memory有优先级，所以需要按照δ的绝对值大小排列
    #除开数据有排列，还需要存储相应的priority的值
    correction=0.01
    #alpha beta都是PER文章中比较重要的参数，决定具体你想要抵消多少优先抽取经验对经验分布的影响
    alpha=0.6
    beta=0.4
    beta_increment=0.0001
    abs_err_upper=1.
    def __init__(self,capacity):
        #self.capacity=capacity
        self.tree=SumTree(capacity)
        
    def store(self,transiton):
        #搜索最大值，只搜索叶子结点
        #这样保证了，新来的片段是有很大概率被优先更新的
        max_priority=np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority==0:
            max_priority=self.abs_err_upper
        self.tree.add(max_priority,transiton)
    
    def sample(self,n):
        b_index,b_memory,IS_weight=np.empty((n,),dtype=np.int32),np.empty((n,self.tree.data[0].size)),np.empty((n,1))
        priority_segment=self.tree.total_p/n
        self.beta=np.min([1.,self.beta+self.beta_increment])
        
        min_prob=np.min(self.tree.tree[-self.tree.capacity:])/self.tree.total_p
        for i in range(n):
            a,b=priority_segment*i, priority_segment*(i+1)
            v=np.random.uniform(a,b)
            index,priority,data=self.tree.get_leaf(v)
            prob=priority/self.tree.total_p
            IS_weight[i,0]=np.power(prob/min_prob,-self.beta)
            b_index[i],b_memory[i,:]=index,data
        return b_index,b_memory,IS_weight
    
    def batch_update(self,tree_index,abs_errors):
        abs_errors+=self.correction
        clipped_errors=np.minimum(abs_errors,self.abs_err_upper) #为什么要设置error的上限呢
        ps=np.power(clipped_errors,self.alpha) #为保证ps值！
        for ti,p in zip(tree_index,ps):
            self.tree.update(ti, p)


class DQN_agent(object):
    def __init__(self,):
        self.env=env          #环境设置为格子世界
        self.pro_fine=None
        self.pro_SandStorm=pro_SandStorm
        self.water=240
        self.food=240
        #用一个很大的负值告诉他不能这样走
        self.reward_tabular=np.array([[-22.5,-67.5,-75],
                                      [-55,-135,-10000],
                                      [-82.5+1000,-202.5+1000,-225+1000]])
        
        self.eval_net, self.target_net = Net(),Net()
        self.learn_step_counter=0         #目标更新，学习步数
        self.memory_counter=0             #存储更新
        self.memory=memory(MEMORY_CAPACITY) #新的记忆模块
        self.optimizer=torch.optim.Adam(self.eval_net.parameters(),lr=LR)
        self.loss_func=nn.MSELoss()
    
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
        
    def store_transition(self,s,a,r,s_):
        #hstack是横向堆栈
        transiton=np.hstack((s,[a,r],s_))
        self.memory_counter+=1
        self.memory.store(transiton)
        
    def learning(self):
         #以TARGET_REPLACE_ITER为频率更新target网络
        if self.learn_step_counter % TARGET_REPLACE_ITER:
            self.target_net.load_state_dict(self.eval_net.state_dict()) #eval参数复制到taget里
        self.learn_step_counter += 1
        
        tree_index, b_memory, IS_weights =self.memory.sample(BATCH_SIZE)
        
        b_s = torch.FloatTensor(b_memory[:, :input_len])
        b_a = torch.LongTensor(b_memory[:, input_len:input_len+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, input_len+1:input_len+2])
        b_s_ = torch.FloatTensor(b_memory[:, -input_len:])
        
        q_eval=self.eval_net(b_s).gather(1,b_a) #即输出b_a这个动作的Q值
        q_next=self.target_net(b_s_).detach()         #detach()保证target_net不会被反向传播
        q_target=b_r+ GAMMA*q_next.max(1)[0].view(BATCH_SIZE,1)  #reshape (batch,1)
        
        TD_error=abs(q_target.clone().detach()-q_eval.clone().detach())
        TD_error=TD_error.numpy()
        self.memory.batch_update(tree_index,TD_error) #更新batch
        IS_weights=torch.FloatTensor(IS_weights)
        loss=self.loss_func(q_eval*IS_weights,q_target*IS_weights) # 没有重要性采样偏置修正
        max_loss=loss.detach().numpy()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return max_loss
    
    def save_model(self,name):
        torch.save(self.target_net.state_dict(),name)
        
    
    def training(self,max_episode_num):
        self._init_agent()
        loss_list=[]
        for num_episode in range(max_episode_num):
            #这个部分可以展示训练进度
            present_percent_0=int((num_episode*100/max_episode_num)*100)/100
            present_percent_1=int(((num_episode+1)*100/max_episode_num)*100)/100
            if present_percent_0!=present_percent_1:
                print("训练进度：{:.2f}%".format(present_percent_0))
            #首先创建一个线性下降的epsilon
            epsilon=max(1-num_episode/max_episode_num,epsilon_lower_bound)
            
            self._init_agent()
            total_reward=0
            self.pro_fine=random.random()
            weather_0=self.get_weather()
            observation_0=self.env.state
            state_0=[observation_0,weather_0,0,self.water,self.food]
            
            present=False
            if num_episode/max_episode_num >0.9999:
                present=True
            
            if present:
                self.env.render()
            
            for i in range(30):
                
                #以更大概率挖矿
                '''
                if observation_0==7:
                    if random.random()<=epsilon:
                        action_0=4
                    else:
                        action_0=int(self.target_net(torch.Tensor(state_0)).argmax())
                else:
                    if random.random()<=epsilon:
                    #选择随机动作
                        action_0=self.env.action_space.sample()
                    else:
                        action_0=int(self.target_net(torch.Tensor(state_0)).argmax())
                '''
                if random.random()<=epsilon:
                #选择随机动作
                    action_0=self.env.action_space.sample()
                else:
                    action_0=int(self.target_net(torch.Tensor(state_0)).argmax())
                
                observation_1,reward,is_done,info=env.step(action_0)
                if present:
                    self.env.render()
                    print("第{}天，到达{}，天气{}".format(i,observation_1,weather_0))
                #两件事：修改reward和自身状态
                reward=self.get_reward(observation_0, observation_1, weather_0)
                
                #修正奖励
                
                if reward <= 0:
                    if reward==-10000:
                        #如果不是沙尘暴
                        reward=-1
                    else:
                        reward=0
                else:
                    reward=1000/1800
                
                
                if is_done:
                    reward+=(self.water*2.5+self.food*5)/1800
                
                if is_done and present:
                    print("到达终点")
                
                if self.water <=0 or self.food <= 0:
                    is_done=True
                    if present:
                        print("未到达终点")
                    #reward -= 10000 #未到达终点的惩罚
                if i == 29 and observation_1!= 4:
                    is_done=True
                    if present:
                        print("未到达终点")
                    #reward -= 10000 #未到达终点的惩罚
                total_reward += reward
                
                weather_1=self.get_weather()
                state_1=[observation_1,weather_1,i+1,self.water,self.food]
                self.store_transition(state_0, action_0, reward, state_1)
                observation_0=observation_1
                state_0=state_1
                weather_0=weather_1
                if is_done:
                    if present:
                        print("本epoch总奖励值为：{}".format(total_reward))
                    break
            if self.memory_counter >= MEMORY_CAPACITY:
                max_loss=float(self.learning())
                loss_list.append(max_loss)
        return loss_list

a=DQN_agent()
Q=a.training(max_episode_num=100000)
y=np.array(Q)
x=np.arange(0,len(y),1)
plt.plot(x, y)
plt.show()
a.save_model("NO4_10w_positive_alpha_10-5.pt")
'''
file = open('NO4_10w_positive_alpha_10-5.pickle', 'wb')
pickle.dump(Q, file)
file.close()
'''


