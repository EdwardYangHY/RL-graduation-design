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
import matplotlib.pyplot as plt


BATCH_SIZE=32
LR=0.00001
#EPSILON=0.9
GAMMA=0.9
TARGET_REPLACE_ITER=100        #目标更新频率/步数
MEMORY_CAPACITY=2000

input_len=5
#输入数据（state，a,r,s',a'）输出一个值

class Net(nn.Module):
    def __init__(self, ):
        super(Net,self).__init__()
        self.fc1=nn.Linear(input_len+1, 50)
        self.fc1.weight.data.normal_(0,0.1) #参数初始化
        self.fc2=nn.Linear(50, 30)
        self.fc2.weight.data.normal_(0,0.1) #参数初始化
        self.out=nn.Linear(30,1)
        self.out.weight.data.normal_(0,0.1) #参数初始化
        #self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        #输入x是一个张量，构成是[state,a]
        #尝试对x进行归一化处理
        #【pos,weather,day,water,food,action】
        #尽量避免0的出现
        if len(x.shape)==1:
            x[0]=(x[0]+1)/14
            x[1]=(x[1]+1)/2
            x[2]=(x[2]+1)/10
            x[3]=x[3]/240
            x[4]=x[4]/240
            x[5]=(x[5]+1)/13
        else:
            x[:,0]=(x[:,0]+1)/14
            x[:,1]=(x[:,1]+1)/2
            x[:,2]=(x[:,2]+1)/10
            x[:,3]=x[:,3]/240
            x[:,4]=x[:,4]/240
            x[:,5]=(x[:,5]+1)/13
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
        self.new_memory=memory(MEMORY_CAPACITY) #新的记忆模块
        self.optimizer=torch.optim.Adam(self.eval_net.parameters(),lr=LR)
        self.loss_func=nn.MSELoss()
    
    def _init_agent(self):
        self.pos=0
        self.state=None
        self.water=240
        self.food=240
        return
    
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
        self.memory_counter+=1
        self.new_memory.store(transiton)
    
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
    
    def choose_max_target(self,state):
        pos_0=int(state[0])
        action=self.get_action(pos_0)
        action_value=torch.zeros(len(action))
        for i in range(len(action)):
            x=np.append(state,[action[i]])
            x=torch.Tensor(x)
            action_value[i]=self.target_net.forward(x)
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
        
    def new_learning(self):
         #以TARGET_REPLACE_ITER为频率更新target网络
        if self.learn_step_counter % TARGET_REPLACE_ITER:
            self.target_net.load_state_dict(self.eval_net.state_dict()) #eval参数复制到taget里
        self.learn_step_counter += 1
        
        tree_index, b_memory, IS_weights =self.new_memory.sample(BATCH_SIZE)
        b_sa=torch.FloatTensor(b_memory[:, :input_len+1])              #构架成功
        b_r = torch.FloatTensor(b_memory[:, input_len+1:input_len+2])
        
        #合并
        #注意这里并不是用choose_max,因为choose max是用动作去评估，应该用target评估
        action_0=self.choose_max_target(b_memory[0, -input_len:])
        action_1=self.choose_max_target(b_memory[1, -input_len:])
        x_0=np.append(b_memory[0, -input_len:],action_0)
        x_1=np.append(b_memory[1, -input_len:],action_1)
        b_s_a=np.stack((x_0,x_1))
        for i in range(2,BATCH_SIZE):
            action=self.choose_max_target(b_memory[i, -input_len:])        #这里不应该是choose_action,应该是最大的动作值
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
        
        for i in range(BATCH_SIZE):
            if int(b_sa[i][2])==9:
                q_target[i]=b_r[i]
            if int(b_sa[i][0])==12:
                q_target[i]=b_r[i]
            if b_sa[i][3]<=0 or b_sa[i][4]<=0:
                q_target[i]=b_r[i]
        
        TD_error=abs(q_target.clone().detach()-q_eval.clone().detach())
        TD_error=TD_error.numpy()
        self.new_memory.batch_update(tree_index,TD_error) #更新batch
        IS_weights=torch.FloatTensor(IS_weights)
        loss=self.loss_func(q_eval*IS_weights,q_target*IS_weights) # 没有重要性采样偏置修正
        max_loss=loss.detach().numpy()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return max_loss
        
        
    def save_model(self,name):
        torch.save(self.target_net.state_dict(),name)
        #如何加载模型
        '''
        model = SampleNet()
        #通过 load_state_dict 函数加载参数，torch.load() 函数中重要的一步是反序列化。
        model.load_state_dict(torch.load("model.pt"))
        '''
   
    def training(self,max_episode_num):
        self._init_agent()
        self.pro_fine=random.random()
        loss_list=[]
        count_mining=0
        for num_episode in range(max_episode_num):
            
            #首先做一个逐步下降但是存在下界的epsilon
            epsilon=1-num_episode/max_episode_num
            #print(epsilon)
            epsilon_lowerbound=0.1
            if epsilon <= epsilon_lowerbound:
                epsilon=epsilon_lowerbound
            
            '''
            #随机更多的epsilon，探索更多，但是收敛更慢
            if num_episode>=max_episode_num*0.99:
                epsilon=0.1
            else:
                epsilon=0.9
            '''
            
            print("训练进度：{:.2f}%".format(num_episode*100/max_episode_num))
            
            self._init_agent()
            total_reward=0
            
            self.pro_fine=random.random()
            
            weather_0=self.get_weather()
            state_0=[self.pos,weather_0,0,self.water,self.food]
            
            for i in range(10):
                pos_0=self.pos #纯展示用
                pos_1=self.choose_action_epsilon(state_0, epsilon)
                #pos_1=self.choose_action_epsilon(state_0, 1)
                if num_episode>=max_episode_num*0.99:
                    print("第{}天，天气{},从{}到{}".format(i,weather_0,pos_0,pos_1))
                pos_1,reward,is_done=self.act(pos_1, weather_0)
                reward=reward/100
                '''
                #具有正奖励的奖励设置
                #对reward也进行归一化处理
                if pos_0==pos_1 and pos_1==8:
                    reward=200/1800
                    count_mining+=1
                elif pos_1==12:
                    reward=(self.water*2.5+self.food*5)/1800
                else:
                    reward=0
                '''
                
                #负奖励，惩罚机制的奖励设置
                if self.water <=0 or self.food <= 0:
                    reward=reward-5
                    is_done=True
                if i==9 and is_done==False:
                    reward=reward-5
                    is_done=True
                
                total_reward+=reward
                
                weather_1=self.get_weather()
                if pos_1==-1:
                    state_1=[8,weather_1,i+1,self.water,self.food]
                else:
                    state_1=[pos_1,weather_1,i+1,self.water,self.food]
                #存储记忆
                self.store_transition(state_0, pos_1, reward, state_1)
                #print(reward)
                #self.store_transition(state_0, pos_1, total_reward, state_1)
                state_0=state_1
                weather_0=weather_1
                
                if is_done:
                    break
            
            if self.memory_counter >= MEMORY_CAPACITY:
                max_loss=float(self.new_learning())
                loss_list.append(max_loss)
        
        return loss_list
                
                

b=DQN()
'''
Q=a.test(pro_fine=0.6,
           lambda_=0.01,
           gamma=0.8,
           alpha=0.1,
           max_episode_num=4000)
'''
#Q_=a.learning()

Q=b.training(max_episode_num=10000)
y=np.array(Q)
x=np.arange(0,len(y),1)
plt.plot(x, y)
plt.show()
#b.save_model("DQN_PER_negetive_1W.pt")



'''
x=[0,0,0,1,1,3]
x=torch.Tensor(x)
hh=a.eval_net.forward(x)

x=np.array([10,0,0,240,240])
b=a.choose_action(x)

'''



