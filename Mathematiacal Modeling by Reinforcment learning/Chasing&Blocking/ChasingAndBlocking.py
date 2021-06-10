# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 09:40:01 2021

@author: Haoyuan Yang

多玩家博弈

使用与毕业设计完全相同的结构去进行

除了reward不同以外其他都一样
"""
import numpy as np
import random
import pickle
import time
import matplotlib.pyplot as plt

chess_map=np.zeros([5,5])

class chess_agent(object):
    def __init__(self):
        self.Map=np.zeros([5,5])
        #按照numpy的想法，左上角是0点
        self.pos_police_1=None
        self.pos_police_2=None
        self.pos_thief=None
        self.pos_all=None #(三个位置)
        self.Q={}
        self.Q_epoch={}
        self.E_epoch={}
        self.action_space=5*5
        
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
        '''
        if action==0:
            print("向上")
        if action==1:
            print("向下")
        if action==2:
            print("向左")
        if action==3:
            print("向右")
        if action==4:
            print("不动")
        '''
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
                return -0.05    #稍微修改一下
        return 1
    
   
    def test_env(self,times):
        count_s=0
        for i in range(times):
            self._init_agent()
            is_done=False
            iteration=0
            #print(self.Map)
            while is_done==False and iteration <=1000:
                #先给小偷随机动作
                action_3=random.randint(0, 4)
                self.thief_act(action_3)
                actions=random.randint(0, 24)
                _state,reward,is_done=self.union_act(actions)
                #print(self.Map)
                if is_done:
                    print("第{}次实验达成目标".format(i))
                    print(self.Map)
                    count_s+=1
                iteration+=1
        print("成功率：{}".format(100*count_s/times))
                
    
    def _assert_state(self,state_name,randomized=True):
        if not self._is_state_in_Q(state_name):
            self._init_state_value(state_name,randomized)
    
    def _is_state_in_Q(self,state_name):
        return self.Q.get(state_name) is not None
    
    def _init_state_value(self,state_name,randomized=True):
        if not self._is_state_in_Q(state_name):
            self.Q[state_name]={}
            for actions in range(self.action_space):
                default_value=random.random() if randomized is True else 0.0
                self.Q[state_name][actions]=default_value
    
    def _get_(self,QorE,state_name,actions):
        self._assert_state(state_name,randomized=True)
        return QorE[state_name][actions]
    
    def _set_(self,QorE,state_name,actions,value):
        self._assert_state(state_name,randomized=True)
        QorE[state_name][actions]=value
    
    def _reset_Q_epoch_E(self):
        self.Q_epoch={}
        self.E_epoch={}
    
    
    def update_Q_epoch(self,theta,alpha):
        #根据参数更新Q_epoch
        for state in self.Q_epoch:
            for actions in self.Q_epoch[state]:
                self.Q_epoch[state][actions]=self.Q_epoch[state][actions]+alpha*theta*self.E_epoch[state][actions]
    
    def update_E_epoch(self,gamma,lambda_):
        #根据参数更新E_epoch
        for state in self.E_epoch:
            for actions in self.E_epoch[state]:
                self.E_epoch[state][actions]=gamma*lambda_*self.E_epoch[state][actions]
    
    def _set_Q_epoch_E(self,state_name):
        #初始化Q_epoch和E_epoch中的值
        #直接从Q表复制
        self.Q_epoch[state_name]={}
        self.E_epoch[state_name]={}
        for action in range(self.action_space): #动作空间
            self.Q_epoch[state_name][action]=self.Q[state_name][action]
            self.E_epoch[state_name][action]=0.0
    
    def update_Q(self):
        #将Q_epoch的值更新到Q里面去
        for state in self.Q_epoch:
            for actions in self.Q_epoch[state]:
                self.Q[state][actions]=self.Q_epoch[state][actions]
                
    def policy(self,state_name,epsilon):
        #新的策略，必须要输入epsilon的值才能继续训练
        self._assert_state(state_name,randomized=True)
        self._set_Q_epoch_E(state_name)
        Q_s=self.Q[state_name]
        rand_value=random.random()
        if rand_value<epsilon:
            #随机生成一个动作
            return random.randint(0, self.action_space-1)
        else:
            return int(max(Q_s,key=Q_s.get))
        
    def get_state(self,action_3):
        #3个坐标+
        state=[]
        for i in range(len(self.pos_all)):
            state.append(self.pos_all[i])
        state.append(action_3)
        return str(state)
        
    def learning(self,lambda_,gamma,alpha,epoch_step,max_episode_num,eval_iteration=10000):
        self.Q={}
        time_start=time.time()
        eval_sr=[]
        eval_ar=[]
        for num_episode in range(max_episode_num):
            if (num_episode % eval_iteration == 0 and num_episode!=0) or num_episode==max_episode_num-1:
                #开始针对averagereturn 做评估
                head='ChasingAndBlocking_Q-learning_'
                mid=str(int(num_episode/eval_iteration)*int(eval_iteration//10000))
                end='w_model.pt'
                file_name=head+mid+end
                file = open(file_name, 'wb')
                pickle.dump(self.Q, file)
                file.close()
                success_rate,average_reward=self.evaluating(1000,epoch_step)
                eval_sr.append(success_rate)
                eval_ar.append(average_reward)
            present_percent_0=int((num_episode*100/max_episode_num)*100)/100
            present_percent_1=int(((num_episode+1)*100/max_episode_num)*100)/100
            if present_percent_0!=present_percent_1:
                print("训练进度：{:.2f}%".format(present_percent_0))
            self._init_agent()
            self._reset_Q_epoch_E()
            
            #构建下降的epsilon
            lower_bound=0.1
            epsilon=num_episode/max_episode_num if num_episode/max_episode_num >= lower_bound else lower_bound
            
            is_done=False
            action_thief=random.randint(0, 4)
            for i in range(epoch_step):
                state=self.get_state(action_thief)
                self.thief_act(action_thief) #已经收集动作，小偷可以行动
                
                union_action=self.policy(state, epsilon)
                pos_all,reward,is_done=self.union_act(union_action)
                
                _action_thief=random.randint(0, 4)
                _state=self.get_state(_action_thief)
                _union_action=self.policy(_state, 0) #贪婪策略选择
                
                Q_error=gamma*self._get_(self.Q, _state, _union_action)-self._get_(self.Q, state, union_action)
                theta=reward+Q_error if not is_done else reward
                E_value=self._get_(self.E_epoch, state, union_action)
                self._set_(self.E_epoch, state, union_action, E_value+1)
                self.update_Q_epoch(theta, alpha)
                self.update_E_epoch(gamma, lambda_)
                action_thief=_action_thief  #比较重要
                if is_done:
                    break
            self.update_Q()
                
        time_end=time.time()
        print("训练总时长{}".format(time_end-time_start))
        return self.Q,eval_sr,eval_ar
            
    def evaluating(self,times,epoch_step):
        print("评估开始")
        tol_success=0
        tol_reward=0
        for num in range(times):
            self._init_agent()
            is_done=False
            action_thief=random.randint(0, 4)
            for i in range(epoch_step):
                state=self.get_state(action_thief)
                self.thief_act(action_thief) #已经收集动作，小偷可以行动
                union_action=self.policy(state,0)
                pos_all,reward,is_done=self.union_act(union_action)
                _action_thief=random.randint(0, 4)
                action_thief=_action_thief  #比较重要
                
                tol_reward+=reward
                if is_done:
                    tol_success += 1
                    print("第{}次评估，成功".format(num))
                    break
        print("评估结束，存活率：{}".format(tol_success/times))
        return tol_success/times,tol_reward/times

'''
a=chess_agent()
#a.test_env(100) 
#随机实验1000步，成功概率17
#随机实验10000步，成功概率97

training_times=1000000
eval_iteration=50000
epoch_step=100
Q,eval_sr,eval_ar=a.learning(lambda_=0.1, gamma=0.8, alpha=0.01,epoch_step=epoch_step, max_episode_num=training_times,eval_iteration=eval_iteration)

head='ChasingAndBlocking_Q-learning_'
mid=str(int(training_times/10000))
end='w_model.pt'
filename=head+mid+end

file = open(filename, 'wb')
pickle.dump(Q, file)
file.close()

y1=np.array(eval_sr)
x1=np.arange(0,len(y1),1)
plt.plot(x1, y1)
plt.show()

y2=np.array(eval_ar)
x2=np.arange(0,len(y2),1)
plt.plot(x2, y2)
plt.show()
'''
