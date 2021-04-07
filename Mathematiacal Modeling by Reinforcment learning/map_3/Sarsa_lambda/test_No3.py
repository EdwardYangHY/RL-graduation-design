# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:13:05 2021

@author: 96174
"""

import No3_Sarsa_Lambda_SearchingModification as NO3
import pickle
import matplotlib.pyplot as plt
import numpy as np

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

file_name='NO3_1500w_negetive_alpha_10-2_return_list.pickle'
plot_average_return(file_name, 100000)