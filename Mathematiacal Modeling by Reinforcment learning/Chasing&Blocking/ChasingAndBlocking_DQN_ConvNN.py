# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 17:17:18 2021

@author: Haoyuan Yang

多玩家博弈
尝试使用卷积神经网络

"""

import numpy as np
import random
import pickle
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

BATCH_SIZE=32
LR=0.00001
#EPSILON=0.9
GAMMA=0.9
TARGET_REPLACE_ITER=100        #目标更新频率/步数
MEMORY_CAPACITY=2000

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3,stride=1,padding=1),nn.ReLU())
        self.conv2=nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,stride=1,padding=1),nn.ReLU())
        
        self.fc1=nn.Sequential(nn.Linear(32*5*5,100),nn.ReLU())
        self.fc2=nn.Sequential(nn.Linear(100, 25))
    
    def forward(self,x):
        x=x.reshape(-1,2,5,5) #数据预处理
        #print(x)
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size()[0],-1)
        x=self.fc1(x)
        x=self.fc2(x)
        return x

class SumTree(object):
    #构建优先经验回放中的组件 Sumtree
    #该数据结构能够极大地减少运算量
    data_pointer=0 #一个指针
    
    def __init__(self,capacity):
        self.capacity=capacity
        self.tree=np.zeros(capacity+capacity-1) #二叉树中，父节点和叶子的关系
                                                #SumTree中存储每个结点的值（子结点的和）
        self.data=np.zeros(capacity,dtype=object)
    
    def add(self,priority,transition):
        tree_index=self.data_pointer+self.capacity-1 #因为前self.capacity-1个index都是父结点
        self.data[self.data_pointer]=transition
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
    def __init__(self):
        self.Map=np.zeros([5,5])
        #按照numpy的想法，左上角是0点
        self.pos_police_1=None
        self.pos_police_2=None
        self.pos_thief=None
        self.pos_all=None #(三个位置)
        self.action_space=5*5
                
        self.eval_net, self.target_net=Net(), Net()
        self.learn_step_counter=0
        self.memory_counter=0
        self.memory=memory(MEMORY_CAPACITY)
        self.optimizer=torch.optim.Adam(self.eval_net.parameters(),lr=LR)
        self.loss_func=nn.MSELoss()
    '''
    def _init_agent(self):
        #随机设置初始位置
        #要保证不重合
        self.Map=np.zeros([5,5])
        self.pos_thief=[random.randint(0, 4),random.randint(0, 4)]
        self.Map[self.pos_thief[0]][self.pos_thief[1]]=3.
        self.pos_police_1=[random.randint(0, 4),random.randint(0, 4)]
        while self.Map[self.pos_police_1[0]][self.pos_police_1[1]] != 0.:
            self.pos_police_1=[random.randint(0, 4),random.randint(0, 4)]
        self.Map[self.pos_police_1[0]][self.pos_police_1[1]]=1.
        self.pos_police_2=[random.randint(0, 4),random.randint(0, 4)]
        while self.Map[self.pos_police_2[0]][self.pos_police_2[1]] != 0.:
            self.pos_police_2=[random.randint(0, 4),random.randint(0, 4)]
        self.Map[self.pos_police_2[0]][self.pos_police_2[1]]=2.
        self.pos_all=[self.pos_police_1[0],self.pos_police_1[1],self.pos_police_2[0],self.pos_police_2[1],self.pos_thief[0],self.pos_thief[1]]
    '''  
    def _init_agent(self):
        #随机设置初始位置
        #要保证不重合
        self.Map=np.zeros([5,5])
        
        self.pos_thief=[0,4]
        self.Map[self.pos_thief[0]][self.pos_thief[1]]=3.
        self.pos_police_1=[0,0]
        self.Map[self.pos_police_1[0]][self.pos_police_1[1]]=1.
        self.pos_police_2=[4,4]
        self.Map[self.pos_police_2[0]][self.pos_police_2[1]]=2.
        self.pos_all=[self.pos_police_1[0],self.pos_police_1[1],self.pos_police_2[0],self.pos_police_2[1],self.pos_thief[0],self.pos_thief[1]]
        
    def get_new_pos(self,old_pos,action):
        #action从0到4，对应 上 下 左 右 不动
        #这是一个三个都可以用的动作值
        
        #这样做是为了防止传引用
        new_pos=[]
        new_pos.append(old_pos[0])
        new_pos.append(old_pos[1])
        
        if action==0:
            #表示向上走
            new_pos[0]-=1
        if action==1:
            #表示向下走
            new_pos[0]+=1
        if action==2:
            #表示向左
            new_pos[1]-=1
        if action==3:
            #表示向右
            new_pos[1]+=1
        if action==4:
            pass
        if new_pos[0] <0 or new_pos[0] >4 or new_pos[1] <0 or new_pos[1] >4:
            return old_pos
        #在判断新的位置上是否空闲
        if self.Map[new_pos[0]][new_pos[1]] != 0:
            return old_pos
        return new_pos
    
    def thief_act(self,action=None):
        if action==None:
            action=random.randint(0,4) #随机选择动作
        new_pos=self.get_new_pos(self.pos_thief,action)
        self.Map[self.pos_thief[0]][self.pos_thief[1]]= 0. #清空原来位置
        self.pos_thief=new_pos
        #self.act(self.pos_thief,action)
        self.Map[self.pos_thief[0]][self.pos_thief[1]]= 3.
        self.pos_all[4],self.pos_all[5]=self.pos_thief[0],self.pos_thief[1]
        
    def union_act(self,actions):
        #这里actions包含了两个动作，police_1 和 police_2的动作
        #返回值 新的状态，奖励和 is_done
        # 1 先动 2 再动
        action_1=actions//5
        action_2=actions%5
        
        new_pos_1=self.get_new_pos(self.pos_police_1, action_1)
        self.Map[self.pos_police_1[0]][self.pos_police_1[1]]= 0. #清空原来位置
        self.pos_police_1=new_pos_1
        self.Map[self.pos_police_1[0]][self.pos_police_1[1]]= 1.
        
        new_pos_2=self.get_new_pos(self.pos_police_2, action_2)
        self.Map[self.pos_police_2[0]][self.pos_police_2[1]]= 0. #清空原来位置
        self.pos_police_2=new_pos_2
        self.Map[self.pos_police_2[0]][self.pos_police_2[1]]= 2.
        
        self.pos_all[0],self.pos_all[1]=self.pos_police_1[0],self.pos_police_1[1]
        self.pos_all[2],self.pos_all[3]=self.pos_police_2[0],self.pos_police_2[1]
        
        reward=self.get_reward()
        is_done = False
        if reward==1:
            is_done = True
        return self.pos_all, reward, is_done
    
    def get_reward(self):
        #怎么样才能获得奖励？thief被围在角落的时候
        for action in range(4):
            #除了停留都试一下
            new_pos=self.get_new_pos(self.pos_thief,action)
            if new_pos!=self.pos_thief:
                return -0.05                 #要不要设置负奖励
        return 1
    
    def store_transition(self,state,union_action,reward,_state):
        #这里可能需要把
        transition=np.hstack((state,[union_action,reward],_state))
        self.memory_counter+=1
        self.memory.store(transition)

        
    def learning(self):
        #数据的处理
        #数据的存储
        if self.learn_step_counter % TARGET_REPLACE_ITER:
            self.target_net.load_state_dict(self.eval_net.state_dict()) #eval参数复制到taget里
        self.learn_step_counter += 1
        
        tree_index, b_memory, IS_weights =self.memory.sample(BATCH_SIZE)
        
        b_s=torch.FloatTensor(b_memory[:, :50])
        b_a=torch.LongTensor(b_memory[:, 50:51].astype(int))
        b_r=torch.FloatTensor(b_memory[:, 51:52])
        b_s_=torch.FloatTensor(b_memory[:, -50:])
        
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
        
    
    def training(self,max_episode_num,eval_iteration=10000,epoch_step=100):
        self._init_agent()
        #loss_list=[]
        start=time.time()
        eval_sr=[]
        for num_episode in range(max_episode_num):
            if (num_episode % eval_iteration == 0 and num_episode!=0) or num_episode==max_episode_num-1:
                #开始针对averagereturn 做评估
                head='ChasingAndBlocking_DQN_ConvNN_'
                mid=str(int(num_episode/eval_iteration)*int(eval_iteration//10000))
                end='w_fixed_model.pt'
                file_name=head+mid+end
                self.save_model(file_name)
                success_rate=self.evaluating(1000,epoch_step)
                eval_sr.append(success_rate)
            #这个部分可以展示训练进度
            present_percent_0=int((num_episode*100/max_episode_num)*100)/100
            present_percent_1=int(((num_episode+1)*100/max_episode_num)*100)/100
            if present_percent_0!=present_percent_1:
                print("训练进度：{:.2f}%".format(present_percent_0))
            #首先创建一个线性下降的epsilon
            lower_bound=0.1
            epsilon=num_episode/max_episode_num if num_episode/max_episode_num >= lower_bound else lower_bound
            
            self._init_agent()
            is_done=False
            action_thief=random.randint(0, 4)
            for i in range(epoch_step):
                data1=self.Map.copy()
                self.thief_act(action_thief) #已经收集动作，小偷可以行动
                data2=self.Map.copy()
                state=np.hstack((data1.reshape(25,),data2.reshape(25,)))
                
                #下降的epsilon策略
                if random.random() <= epsilon:
                    union_action=random.randint(0,24)
                else:
                    union_action=int(self.target_net(torch.FloatTensor(state)).argmax())
                
                #union_action=int(self.target_net(torch.FloatTensor(state)).argmax())
                #print(union_action)
                pos_all,reward,is_done=self.union_act(union_action)
                
                _action_thief=random.randint(0, 4)
                data3=self.Map.copy()
                self.thief_act(action_thief) #已经收集动作，小偷可以行动
                data4=self.Map.copy()
                _state=np.hstack((data3.reshape(25,),data4.reshape(25,)))
                
                self.store_transition(state,union_action,reward,_state)
                
                action_thief=_action_thief  #比较重要
                if is_done:
                    #print("耗费{}步追到小偷".format(i))
                    break
            if self.memory_counter >= MEMORY_CAPACITY:
                self.learning()
            
        end=time.time()
        print("运行总时间{}".format(start-end))
        return eval_sr
    
    def evaluating(self,times,epoch_step):
        print("评估开始")
        tol_success=0
        for num in range(times):
            self._init_agent()
            is_done=False
            action_thief=random.randint(0, 4)
            for i in range(epoch_step):
                data1=self.Map.copy()
                self.thief_act(action_thief) #已经收集动作，小偷可以行动
                data2=self.Map.copy()
                state=np.hstack((data1.reshape(25,),data2.reshape(25,)))
                
                union_action=int(self.target_net(torch.FloatTensor(state)).detach().argmax())
                pos_all,reward,is_done=self.union_act(union_action)
                
                _action_thief=random.randint(0, 4)
                data3=self.Map.copy()
                self.thief_act(action_thief) #已经收集动作，小偷可以行动
                data4=self.Map.copy()
                _state=np.hstack((data3.reshape(25,),data4.reshape(25,)))
                
                self.store_transition(state,union_action,reward,_state)
                
                action_thief=_action_thief  #比较重要
                if is_done:
                    tol_success += 1
                    print("第{}次评估，成功".format(num))
                    break
        print("评估结束，存活率：{}".format(tol_success/times))
        return tol_success/times
    
    def evaluating_random(self,times):
        print("评估开始")
        tol_success=0
        for num in range(times):
            self._init_agent()
            is_done=False
            action_thief=random.randint(0, 4)
            for i in range(100):
                data1=self.Map.copy()
                self.thief_act(action_thief) #已经收集动作，小偷可以行动
                data2=self.Map.copy()
                state=np.hstack((data1.reshape(25,),data2.reshape(25,)))
                
                union_action=random.randint(0, 24)
                pos_all,reward,is_done=self.union_act(union_action)
                
                _action_thief=random.randint(0, 4)
                data3=self.Map.copy()
                self.thief_act(action_thief) #已经收集动作，小偷可以行动
                data4=self.Map.copy()
                _state=np.hstack((data3.reshape(25,),data4.reshape(25,)))
                
                self.store_transition(state,union_action,reward,_state)
                
                action_thief=_action_thief  #比较重要
                if is_done:
                    tol_success += 1
                    #print("第{}次评估，成功".format(num))
                    break
        print("评估结束，存活率：{}".format(tol_success/times))
        return tol_success/times
    
    def save_model(self,name):
        torch.save(self.target_net.state_dict(),name)

a=DQN_agent()

training_times=3000000
evaluating_iterations=100
epoch_step=30
eval_sr=a.training(training_times,eval_iteration=evaluating_iterations,epoch_step=epoch_step)

head='ChasingAndBlocking_DQN_ConvNN_'
mid=str(int(training_times/10000))
end='w_fixed_model.pt'
file_name=head+mid+end
a.save_model(file_name)

y=np.array(eval_sr)
x=np.arange(0,len(y),1)
plt.plot(x, y)
plt.show()
