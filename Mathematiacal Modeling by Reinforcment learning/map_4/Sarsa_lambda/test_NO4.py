# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:59:15 2021

@author: 96174
"""

import NO_4_sarsa_SeachingModification as NO4
import pickle
import matplotlib.pyplot as plt
import numpy as np

'''
file_name='NO4_2000w_negetive_alpha_10-2.pickle'
a=NO4.Sarsa_Agent()
a.show_policy(file_name,10)
'''

file_name='NO4_1000w_negetive_alpha_10-2.pickle'


def plot_average_return(file_name,iteration):
    with open(file_name, 'rb') as file:
        ar_list=pickle.load(file)
    
    count_num=0
    count_sum=0
    #iteration=10
    new_list=[]
    for i in range(len(ar_list)):
        count_num+=1
        count_sum+=ar_list[i]
        if i%iteration ==0 or i==len(ar_list)-1:
            sum_ar=count_sum/count_num
            new_list.append(sum_ar)
            count_num=0
            count_sum=0
    
    
    y=np.array(new_list)
    x=np.arange(0,len(y),1)
    plt.plot(x, y)
    plt.show()
    return

def present(Q_name,test_times):
    a=NO4.Sarsa_Agent()
    a.show_policy(Q_name,test_times)
    return

#plot_average_return(file_name,100)
#present(file_name, 10)

file_1000w='NO4_1000w_negetive_alpha_10-2.pickle'
file_2000W='NO4_2000w_negetive_alpha_10-2.pickle'

a=NO4.Sarsa_Agent()
ar1000w,rt1=a.evaluate_policy(10000,file_1000w)
ar2000w,rt2=a.evaluate_policy(10000,file_2000W)