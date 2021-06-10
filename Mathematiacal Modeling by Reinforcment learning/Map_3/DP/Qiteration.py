# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 22:14:55 2021

@author: Edward

动态规划方法 Q值迭代
"""
#用动态规划+高斯赛达尔迭代
#s=[pos,weather,day,water,food]

import random
import numpy as np
from No3_map import Map
import pickle
import time
import matplotlib.pyplot as plt
import gym
env=gym.make('GridWorld-v3')

class sarsa_agent(object):
    def __init__(self,Map):
        self.Map=Map
        self.Q={}
        self.pro_fine=0.5   #简化模型，将其变成0.5
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
    
    def get_reward(self,pos_0,pos_1,weather,water,food):
        new_water=water
        new_food=food
        #是否需要对到达终点的行为进行奖励
        if pos_0 >=13 or pos_1 >= 13:
            print("超出地图")
            return None
        if weather==0:
            consume=-(2.5*3+5*4)
            consume_water=3
            consume_food=4
        else:
            consume=-(2.5*9+5*9)
            consume_water=9
            consume_food=9
        
        #####
        if pos_0==pos_1 or pos_1==-1:
            #说明没有移动,需要判断能否挖矿
            #改进以后（8，8）变成了停留，（8，-1）变成了挖矿
            if pos_1==-1:
                #能挖矿
                consume=consume*3
                new_water=water-consume_water*3
                new_food=food-consume_food*3
                reward=consume+200
                return reward,new_water,new_food
            else:
                #不能挖矿
                reward=consume
                new_water=water-consume_water
                new_food=food-consume_food
                return reward,new_water,new_food
         #####   
            
        else:
            #说明移动了
            consume=consume*2
            new_water=water-consume_water*2
            new_food=food-consume_food*2
            reward=consume
            return reward,new_water,new_food
    
    def state_transfer(self,state_name,action):
        #输入s 和 a，得到所有可能的下个状态和其对应的概率
        #下一个状态只有两个，概率取决于模型的天气参数
        #所以最重要的是天数，食物和水的变化
        is_done=False
        state_0=eval(state_name)
        pos_0=state_0[0]
        weather_0=state_0[1]
        water_0=state_0[3]
        food_0=state_0[4]
        
        if state_0[2]!=9:
            day_1=state_0[2]+1
        else:
            #天数到达上限，结束
            day_1=9
            is_done=True
        
        reward_1,water_1,food_1=self.get_reward(pos_0,action,weather_0,water_0,food_0)
        
        #中途资源耗尽，结束
        if water_1<0 or food_1<0:
            is_done=True
        
        if action!=-1:
            pos_1=action
        else:
            pos_1=8
        
        #到达终点时，结束
        if pos_1==12:
            is_done=True
            
        #如果结束时，不在最终的位置
        if is_done and pos_1 != 12:
            reward_1-=1000
            
        #这里有点小问题，下个状态的值是由当天的天气决定的跟下一天没有关系
        state_1_0=str([pos_1,0,day_1,water_1,food_1])
        state_1_1=str([pos_1,1,day_1,water_1,food_1])
        
        return state_1_0, state_1_1, reward_1, is_done
    
    def _assert_state(self,state_name,randomized=True):
        if not self._is_state_in_Q(state_name):
            self._init_state_value(state_name,randomized)
    
    def _is_state_in_Q(self,state_name):
        return self.Q.get(state_name) is not None
    
    def _init_state_value(self,state_name,randomized=True):
        if not self._is_state_in_Q(state_name):
            self.Q[state_name]={}
            state_list=eval(state_name)
            action_space=self.get_action(state_list[0]) #获取状态空间
            for pos_1 in action_space:
                default_value=random.random() if randomized is True else 0.0
                self.Q[state_name][pos_1]=default_value
        return
    
    def _get_(self,QorE,state_name,pos_1):
        self._assert_state(state_name,randomized=True)
        return QorE[state_name][pos_1]
    
    def _set_(self,QorE,state_name,pos_1,value):
        self._assert_state(state_name,randomized=True)
        QorE[state_name][pos_1]=value
        
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
    
    def update_Q(self):
        pass
    
    #policy也需要做相应的修改
    def learning(self,gamma=0.9,max_iterations=10000,eval_iteration=1000):
        self.Q={}
        state_0=str([0,0,0,self.water,self.food])
        state_1=str([0,1,0,self.water,self.food])
        self._init_state_value(state_0,True)
        self._init_state_value(state_1,True)
        
        eval_ar=[]
        eval_sr=[]
        start_time=time.time()
        for num in range(max_iterations):
            #eval_iteration=1000000
            if num % eval_iteration == 0 or num==max_iterations-1:
                #开始针对averagereturn 做评估
                head='NO3_'
                mid=str(int(num/eval_iteration)*int(eval_iteration//1000))
                end='k_DP_QIteration.pickle'
                file_name=head+mid+end
                file = open(file_name, 'wb')
                pickle.dump(self.Q, file)
                file.close()
                average_return,success_rate=self.evaluate_policy(1000)
                print("平均收益率{}".format(average_return))
                print("平均生存率{}".format(success_rate))
                eval_ar.append(average_return)
                eval_sr.append(success_rate)
            present_percent_0=int((num*100/max_iterations)*100)/100
            present_percent_1=int(((num+1)*100/max_iterations)*100)/100
            if present_percent_0!=present_percent_1:
                print("训练进度：{:.2f}%".format(present_percent_0))
            Q_now=self.Q.copy()
            #print(Q_now)
            #time.sleep(5)
            for state in Q_now:
                #这个循环中，size会变化，所以还是做原来的方法：给两个Q表
                for action in self.Q[state]:
                    #对每一个状态动作对都更新
                    #_state_0,reward_0,is_done_0,_state_1,reward_1,is_done_1 = self.state_transfer(state, action)
                    _state_0, _state_1, reward, is_done = self.state_transfer(state, action)
                    
                    if is_done:
                        value_0=0
                        value_1=0
                    else:
                        action_0=self.new_policy(_state_0, 0)
                        value_0=gamma*self._get_(self.Q, _state_0, action_0)
                        action_1=self.new_policy(_state_1, 0)
                        value_1=gamma*self._get_(self.Q, _state_1, action_1)
                    
                    new_Q_value=self.pro_fine*(reward+value_0)+(1-self.pro_fine)*(reward+value_1)
                    self._set_(self.Q, state, action, new_Q_value)
        end_time=time.time()
        print('用时:{}'.format(end_time-start_time))
        return self.Q,eval_ar,eval_sr
    
    def act(self,pos_1,weather):
        #此时act是无法改变自身状态（self.state）的，因为状态需要天气和天数
        is_done=False
        pos_0=self.pos
        #如果挖矿那么自身状态不改变
        if pos_1!=-1:
            self.pos=pos_1
        
        reward=self.get_reward(pos_0, pos_1, weather,240,240)
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
    
    def evaluate_policy(self,times,Q_name=None):
        if Q_name is not None:
            with open(Q_name, 'rb') as file:
                Q=pickle.load(file)
                self.Q=Q
        average_return=0
        success_count=0
        for num in range(times):
            self._init_agent()
            #self.pro_fine=random.random()
            total_reward=0
            for i in range(30):
                pos_0=self.pos
                weather_0=self.get_weather()
                water=int(self.water/240*10)
                food=int(self.food/240*10)
                state_0=str([pos_0,weather_0,i,water,food])
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

if __name__=='__main__':
    a=sarsa_agent(Map)
    training_times=10000
    eval_iteration=1000
    Q,ar,sr=a.learning(gamma=0.9,max_iterations=training_times,eval_iteration=eval_iteration)
    
    y=np.array(ar)
    x=np.arange(0,len(y),1)
    plt.plot(x, y)
    plt.show()
    
    y1=np.array(sr)
    x1=np.arange(0,len(y1),1)
    plt.plot(x1, y1)
    plt.show()
    
    
    head='NO_3_'
    mid=str(int(training_times/1000))
    end='k_DP_QIteration'
    tail_1='.pickle'
    #tail_2='_return_list.pickle'
    tail_3='_eval_ar.pickle'
    tail_4='_eval_sr.pickle'
    
    
    file = open(head+mid+end+tail_1, 'wb')
    pickle.dump(Q, file)
    file.close()
    
    file = open(head+mid+end+tail_3, 'wb')
    pickle.dump(ar, file)
    file.close()
    
    file = open(head+mid+end+tail_4, 'wb')
    pickle.dump(sr, file)
    file.close()



