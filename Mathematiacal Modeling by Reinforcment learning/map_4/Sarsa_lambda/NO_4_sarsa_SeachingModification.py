# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:12:02 2021

@author: Haoyuan Yang

毕业设计：穿越沙漠 第二题 第四关 格子世界环境
"""
#负向奖励版本
#优化更新过程，理论上能减少很大的运算量
import gym
import random
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

pro_SandStorm=0.1
epsilon_lower_bound=0.1

env=gym.make('GrandGridWorld-v0')
env=env.DessertCrossing()
#env=env.DessertCrossing_simple() #一个简化过后的环境，为了保证训练的效率
env.reset()

#在整个DessertCrossing环境中，动作空间是分离的
#0：左  1：右  2：上  3：下  4：不动
#state=x+n_width*y  坐标轴原点在左下角

class Sarsa_Agent(object):
    def __init__(self,):
        self.env=env          #环境设置为格子世界
        self.Q={}         #用来存所有的状态动作索引
        self.Q_epoch={}   #用来暂存一个epoch状态的索引
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
        #这个版本把E表从初值表中剥离出去了，因为E是随着Q_epoch更新的
        if not self._is_state_in_Q(state_name):
            self.Q[state_name]={}
            for action in range(self.env.action_space.n):
                default_v=random.random()/10 if randomized is True else 0.0
                self.Q[state_name][action]=default_v
        return
    
    '''
    def E_copy_Q(self):
        #这个函数是为了reload 准备的
        #如果需要reload训练好的模型（Q表），则需要复制一下
        for state in self.Q:
            self.E[state]={}
            for action in range(self.env.action_space.n):
                self.E[state][action]=0.0
    '''
    
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
        for action in range(self.env.action_space.n):
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
    
                
    def policy(self,state_name,epsilon):
        self._assert_state(state_name,randomized=False)     #最好不要随机初始化
        self._set_Q_epoch_E(state_name)                     #设置Q_epoch E
        Q_s=self.Q_epoch[state_name]                        #改成Q_epoch 减少索引次数
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
        loss_list=[]
        return_list=[]
        average_return=0
        return_iteration=1000
        
        #用于评估 average——return 和成功率的list
        eval_ar=[]
        eval_sr=[]
        for num_episode in range(max_episode_num):
            eval_iteration=10000
            if num_episode % eval_iteration == 0 or num_episode==max_episode_num-1:
                #开始针对averagereturn 做评估
                head='NO4_'
                mid=str(int(num_episode/eval_iteration)*int(eval_iteration//10000))
                end='w_automatic_alpha_10-2.pickle'
                file_name=head+mid+end
                file = open(file_name, 'wb')
                pickle.dump(self.Q, file)
                file.close()
                average_return,success_rate=self.evaluate_policy(10000)
                eval_ar.append(average_return)
                eval_sr.append(success_rate)
                
            #这个部分可以展示训练进度
            present_percent_0=int((num_episode*100/max_episode_num)*100)/100
            present_percent_1=int(((num_episode+1)*100/max_episode_num)*100)/100
            if present_percent_0!=present_percent_1:
                print("训练进度：{:.2f}%".format(present_percent_0))
            #首先创建一个线性下降的epsilon
            epsilon=max(1-num_episode/max_episode_num,epsilon_lower_bound)
            
            self._init_agent()
            #self._reset_E()
            self._reset_Q_epoch_E()
            total_reward=0
            self.pro_fine=random.random()
            weather_0=self.get_weather()
            observation_0=self.env.state
            water=int(self.water/240*10)
            food=int(self.food/240*10)
            state_0=self._name_state([observation_0,weather_0,0,water,food])
            
            present=False
            if num_episode/max_episode_num >0.999999:
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
                #print("Q表长度{},E表长度{}".format(len(self.Q),len(self.E)))
                
                self.update_Q_epoch(theta, alpha)
                self.update_E(gamma, lambda_)
                observation_0=observation_1
                weather_0=weather_1
                state_0=state_1
                if is_done:
                    #计算平均返回值
                    average_return+=total_reward
                    #计算平均返回值
                    if present:
                        print("总奖励值：{}".format(total_reward))
                    break
            error_epoch=self.update_Q()
            loss_list.append(error_epoch)
            time_end=time.time()
            
            #计算一下average return（外部处理，看下面）
            '''
            if num_episode%return_iteration == 0 and num_episode!=0:
                return_list.append(average_return/return_iteration)
                average_return=0
            '''
            #外部处理iteration的数据
            if num_episode!=0:
                return_list.append(total_reward)
            
        print("训练总时间{}".format(time_end-time_start))
        return self.Q, return_list, eval_ar, eval_sr
         
    def show_policy(self,Q_name,times):
        with open(Q_name, 'rb') as file:
            Q=pickle.load(file)
        
        self.env.render()
        time.sleep(5)
        for num in range(times):
            print("第{}次测试".format(num+1))
            self._init_agent()
            self.Q=Q
            self.pro_fine=random.random()
            total_reward=0
            for i in range(30):
                observation_0=self.env.state
                weather_0=self.get_weather()
                water=int(self.water/240*10)
                food=int(self.food/240*10)
                state_0=self._name_state([observation_0,weather_0,i,water,food])
                action_0=self.policy(state_0, 0.01)
                if weather_0==2:
                    #沙暴天气强制不能移动
                    action_0=4
                observation_1,reward,is_done,info=env.step(action_0)
                reward=self.get_reward(observation_0, observation_1, weather_0)
                if reward >=0:
                    reward=1000
                elif reward==-1620:
                    reward=-1620
                else:
                    reward=0
                env.render()
                if weather_0==0:
                    print("第{}天，到达位置{}，天气晴朗，获得奖励{}".format(i,self.env.state,reward))
                if weather_0==1:
                    print("第{}天，到达位置{}，天气炎热，获得奖励{}".format(i,self.env.state,reward))
                if weather_0==2:
                    print("第{}天，到达位置{}，沙尘暴，获得奖励{}".format(i,self.env.state,reward))
                
                time.sleep(0.3)
                total_reward+=reward
                if self.food <= 0 or self.water <=0:
                    print("因为资源紧缺半路阵亡")
                    break
                
                if i==29 and not is_done:
                    print("规定时间未到达")
                    break
                
                if is_done:
                    final_money=6400+self.water*2.5+self.food*5+total_reward
                    print("到达终点，用时共{}天，剩余水{}，剩余食物{}，最后资金{}".format(i,self.water,self.food,final_money))
                    break
        
        time.sleep(3)
        self.env.close()
        
    
    def evaluate_policy(self,times,Q_name=None):
        #既可以评估当前策略，也可以评估存为pickle的策略
        if Q_name is not None:
            with open(Q_name, 'rb') as file:
                Q=pickle.load(file)
                self.Q=Q
        average_return=0
        success_count=0
        for num in range(times):
            '''
            present_percent_0=int((num*100/times)*100)/100
            present_percent_1=int(((num+1)*100/times)*100)/100
            if present_percent_0!=present_percent_1:
                print("测试进度：{:.2f}%".format(present_percent_0))
            '''
            self._init_agent()
            self.pro_fine=random.random()
            total_reward=0
            for i in range(30):
                observation_0=self.env.state
                weather_0=self.get_weather()
                water=int(self.water/240*10)
                food=int(self.food/240*10)
                state_0=self._name_state([observation_0,weather_0,i,water,food])
                action_0=self.policy(state_0, 0)
                if weather_0==2:
                    #沙暴天气强制不能移动
                    action_0=4
                observation_1,reward,is_done,info=env.step(action_0)
                reward=self.get_reward(observation_0, observation_1, weather_0)
                
                if reward >=0:
                    reward=1000
                elif reward==-1620:
                    reward=-1620
                else:
                    reward=0
                
                total_reward+=reward
                if self.food <= 0 or self.water <=0:
                    total_reward=0
                    break
                
                if i==29 and not is_done:
                    total_reward=0
                    break
                
                if is_done:
                    total_reward=6400+self.water*2.5+self.food*5+total_reward
                    success_count+=1
                    break
            average_return+=total_reward/times
        return average_return,success_count/times
    


training_times=100000

a=Sarsa_Agent()
Q,loss_list,ar,sr=a.learning(lambda_=0.1, 
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


head='NO_4_'
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
pickle.dump(loss_list, file)
file.close()

file = open(head+mid+end+tail_3, 'wb')
pickle.dump(ar, file)
file.close()

file = open(head+mid+end+tail_4, 'wb')
pickle.dump(sr, file)
file.close()