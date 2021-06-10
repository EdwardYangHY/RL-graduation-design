# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 19:27:40 2021

@author: Haoyuan Yang

The mathematical modeling contest in 2020

Map NO.3

Reinforcement Learning in Q-learning （off-policy）

"""
#本文件在天气完全随机的概率下训练，
#决策考虑 当天天气，当前位置，剩余的水和食物
#考虑到尽可能减少状态空间，水和食物在扣除后会以一个十以内的整数存储在“状态里”，当水和食物小于0时会触发惩罚
#目前最完善的基于表值的强化学习方法

#基于version_1:改进挖矿系统,即可以选择挖矿也可以选择不挖
#针对Q值更新做一个优化


import random
import numpy as np
from No3_map import Map
import pickle
import time
import matplotlib.pyplot as plt

class sarsa_agent(object):
    def __init__(self,Map):
        self.Map=Map
        self.Q={}
        self.Q_epoch={}
        self.E={}
        self.pro_fine=None
        self.pos=0
        self.state=None
        self.total_reward=0
        self.water=240
        self.food=240
        return
    
    def _init_agent(self):
        self.pos=0
        self.state=None
        self.water=240
        self.food=240
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
        if pos_0==8:
            #说明到达矿山，此时给出动作挖矿
            action.append(-1)
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
    
    def _reset_Q_epoch_E(self):
        #重新赋值为0，千万不要初始化
        #现在可以初始化了
        '''
        for value_dic in self.E.values():
            for action in range(self.env.action_space.n):
                value_dic[action]=0.00
        '''
        self.Q_epoch={}
        self.E={}
    
    
    def update_Q_epoch(self,theta,alpha):
        #根据参数更新Q_epoch
        for state in self.Q_epoch:
            for pos_1 in self.Q_epoch[state]:
                self.Q_epoch[state][pos_1]=self.Q_epoch[state][pos_1]+alpha*theta*self.E[state][pos_1]
    
    def update_E(self,gamma,lambda_):
        #根据参数更新E
        for state in self.E:
            for pos_1 in self.E[state]:
                self.E[state][pos_1]=gamma*lambda_*self.E[state][pos_1]
    
    def _set_Q_epoch_E(self,state_name):
        #初始化Q_epoch和E_epoch中的值
        #直接从Q表复制
        self.Q_epoch[state_name]={}
        self.E[state_name]={}
        pos_0=int(eval(state_name)[0])
        for action in self.get_action(pos_0): #动作空间
            self.Q_epoch[state_name][action]=self.Q[state_name][action]
            self.E[state_name][action]=0.0
        pass
    
    def update_Q(self):
        #将Q_epoch的值更新到Q里面去
        error=0
        for state in self.Q_epoch:
            for pos_1 in self.Q_epoch[state]:
                if abs(self.Q[state][pos_1]-self.Q_epoch[state][pos_1]) >= error:
                    error=abs(self.Q[state][pos_1]-self.Q_epoch[state][pos_1])
                self.Q[state][pos_1]=self.Q_epoch[state][pos_1]
        return error


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
        self._assert_state(state_name,randomized=False)     #最好不要随机初始化
        self._set_Q_epoch_E(state_name)                     #设置Q_epoch E
        Q_s=self.Q_epoch[state_name]
        rand_value=random.random()
        if use_epsilon and rand_value<epsilon:
            #随机生成一个动作
            pos_0=eval(state_name)[0] #获取当前位置
            action=self.get_action(pos_0) #获取动作空间
            ra=random.randint(0, len(action)-1)
            return action[ra]
        else:
            return int(max(Q_s,key=Q_s.get))
        
    def new_policy(self,state_name,epsilon):
        #新的策略，必须要输入epsilon的值才能继续训练
        self._assert_state(state_name,randomized=True)
        Q_s=self.Q[state_name]
        rand_value=random.random()
        if rand_value<epsilon:
            #随机生成一个动作
            pos_0=eval(state_name)[0] #获取当前位置
            action=self.get_action(pos_0) #获取动作空间
            ra=random.randint(0, len(action)-1)
            return action[ra]
        else:
            return int(max(Q_s,key=Q_s.get))
                    
    
    def learning(self,lambda_,gamma,alpha,max_episode_num):
        self.Q={}
        self.E={}
        self._init_agent()
        time_start=time.time()
        return_list=[]
        average_return=0
        
        eval_ar=[]
        eval_sr=[]
        for num_episode in range(max_episode_num):
            eval_iteration=1000000
            if num_episode % eval_iteration == 0 or num_episode==max_episode_num-1:
                #开始针对averagereturn 做评估
                head='NO3_'
                mid=str(int(num_episode/eval_iteration)*int(eval_iteration//10000))
                end='w_automatic_alpha_10-2.pickle'
                file_name=head+mid+end
                file = open(file_name, 'wb')
                pickle.dump(self.Q, file)
                file.close()
                average_return,success_rate=self.evaluate_policy(10000)
                eval_ar.append(average_return)
                eval_sr.append(success_rate)
            
            present_percent_0=int((num_episode*100/max_episode_num)*100)/100
            present_percent_1=int(((num_episode+1)*100/max_episode_num)*100)/100
            if present_percent_0!=present_percent_1:
                print("训练进度：{:.2f}%".format(present_percent_0))
            
            
            self._init_agent()
            self._reset_Q_epoch_E()
            total_reward=0
            
            #不再对天气的概率做约束，是否就是我每次随机生成一组天气然后训练呢
            #新代码
            self.pro_fine=random.random() #这样每次生成都是随机的
            #新代码
            
            water=int(self.water/240*10)
            food=int(self.food/240*10)
            
            weather_0=self.get_weather()
            state_0=[self.pos,weather_0,0,water,food]
            #pos_0=self.pos
            
            present=False
            if num_episode/max_episode_num >0.999999:
                present=True
            
            
            for i in range(10):
                
                #print("剩余水{}，剩余食物{}".format(water,food))
                
                pos_1=self.policy(str(state_0), num_episode, max_episode_num, use_epsilon=True) #这里是行动策略，epsilon-greedy
                if present:
                    print("第{}天，从{}到{}，天气{}".format(i,self.pos,pos_1,weather_0))
                pos_1,reward,is_done=self.act(pos_1, weather_0)
                
                water=int(self.water/240*10)
                food=int(self.food/240*10)
                
                #还要判断是否花光了补给！
                if self.water<0 or self.food<0:
                    reward=reward-1000
                    #注意：想要展现出效果，需要将这个-1000的反馈值传递给Q表和E表！
                    theta=reward  #到达终点后没有下一个状态了，所以直接用reward传值给theta
                    E_value=self._get_(self.E, str(state_0), pos_1)
                    self._set_(self.E, str(state_0), pos_1, E_value+1)
                    self.update_Q_epoch(theta, alpha)
                    self.update_E(gamma, lambda_)
                    total_reward+=reward
                    if present:
                        print("补给用完，总奖励：{:.2f}".format(total_reward))
                    break
                
                if i==9 and is_done==False:
                    reward=reward-1000
                    #注意：想要展现出效果，需要将这个-1000的反馈值传递给Q表和E表！
                    theta=reward  #到达终点后没有下一个状态了，所以直接用reward传值给theta
                    E_value=self._get_(self.E, str(state_0), pos_1)
                    self._set_(self.E, str(state_0), pos_1, E_value+1)
                    self.update_Q_epoch(theta, alpha)
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
                #这个状态列表里不能有-1
                if pos_1==-1:
                    state_1=[8,weather_1,i+1,water,food]   #下一个时刻的状态S'
                else:
                    state_1=[pos_1,weather_1,i+1,water,food]   #下一个时刻的状态S'
                
                
                pos_2=self.policy(str(state_1), num_episode, max_episode_num, use_epsilon=False) #目标更新，greedy
                theta=reward+gamma*self._get_(self.Q, str(state_1), pos_2)-self._get_(self.Q, str(state_0), pos_1)
                E_value=self._get_(self.E, str(state_0), pos_1)
                self._set_(self.E, str(state_0), pos_1, E_value+1)
                self.update_Q_epoch(theta, alpha)
                self.update_E(gamma, lambda_)
                '''
                print('Q表的更新值'.format(self.Q_epoch))
                print('E表的更新值'.format(self.E))
                '''
                print(self.Q_epoch)
                #pos_0=pos_1
                #pos_1=pos_2          #更新动作A
                weather_0=weather_1  #更新天气
                state_0=state_1      #更新状态S
            
            self.update_Q()
            if num_episode!=0:
                return_list.append(total_reward)
            time_end=time.time()
        print("训练总时间{}".format(time_end-time_start))
        return self.Q,return_list,eval_ar,eval_sr
    
    
    
    def reload(self,Q_np_name):
        #这里的reload重载只是展示
        with open(Q_np_name, 'rb') as file:
            Q=pickle.load(file)
        self._init_agent()
        self.Q=Q
        self.pro_fine=random.random()
        pos_0=self.pos
        water=int(self.water/240*10)
        food=int(self.food/240*10)
        weather_0=self.get_weather()
        state_0=[pos_0,weather_0,0,water,food]
        pos_1=self.policy(str(state_0), 0, 10, use_epsilon=False)
        for i in range(1,11):
            pos_1,reward,is_done=self.act(pos_1, weather_0)
            print("第{}天，从{}到{}，天气{}".format(i,pos_0,pos_1,weather_0))
            if is_done:
                #print(i)
                break
            water=int(self.water/240*10)
            food=int(self.food/240*10)
            weather_0=self.get_weather()
            state_0=[self.pos,weather_0,i,water,food]
            pos_0=self.pos
            pos_1=self.policy(str(state_0), 0, 10, use_epsilon=False)
            
    def evaluate_policy(self,times,Q_name=None):
        if Q_name is not None:
            with open(Q_name, 'rb') as file:
                Q=pickle.load(file)
                self.Q=Q
        average_return=0
        success_count=0
        for num in range(times):
            self._init_agent()
            self.pro_fine=random.random()
            total_reward=0
            for i in range(30):
                pos_0=self.pos
                weather_0=self.get_weather()
                water=int(self.water/240*10)
                food=int(self.food/240*10)
                state_0=self._name_state([pos_0,weather_0,i,water,food])
                pos_1=self.new_policy(state_0,0)
                pos_1,reward,is_done=self.act(pos_1, weather_0)
                
                if reward==117.5 or reward==-2.5:
                    reward=200
                else:
                    reward=0
                total_reward+=reward
                
                if self.food < 0 or self.water <0:
                    total_reward=0
                    break
                
                if i==9 and not is_done:
                    total_reward=0
                    break
                
                if is_done:
                    total_reward=6400+self.water*2.5+self.food*5+total_reward
                    success_count+=1
                    break
            average_return+=total_reward/times
        return average_return,success_count/times



a=sarsa_agent(Map)

training_times=15000000

Q,return_list,ar,sr=a.learning(lambda_=0.1,
           gamma=0.8,
           alpha=0.01,
           max_episode_num=training_times)

y=np.array(ar)
x=np.arange(0,len(y),1)
plt.plot(x, y)
plt.show()

y1=np.array(sr)
x1=np.arange(0,len(y1),1)
plt.plot(x1, y1)
plt.show()


head='NO_3_'
mid=str(int(training_times/10000))
end='w_automatic_alpha_10-2'
tail_1='.pickle'
tail_2='_return_list.pickle'
tail_3='_eval_ar.pickle'
tail_4='_eval_sr.pickle'


file = open(head+mid+end+tail_1, 'wb')
pickle.dump(Q, file)
file.close()

file = open(head+mid+end+tail_2, 'wb')
pickle.dump(return_list, file)
file.close()

file = open(head+mid+end+tail_3, 'wb')
pickle.dump(ar, file)
file.close()

file = open(head+mid+end+tail_4, 'wb')
pickle.dump(sr, file)
file.close()



