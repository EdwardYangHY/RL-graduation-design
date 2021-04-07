# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 19:11:08 2021

@author: Haoyuan Yang

The mathematical modeling contest in 2020

Map NO.3

Reinforcement Learning in SARSA

"""

import numpy as np

Map=np.eye(13)

def setedge(Map,vertex1,vertex2):
    m,n=Map.shape
    vertex1=vertex1-1
    vertex2=vertex2-1
    if vertex1 > m or vertex2 > m:
        print("Valid vertex information")
        return None
    Map[vertex1,vertex2]=1
    Map[vertex2,vertex1]=1
    return Map
    
def setarc(Map,vertex1,vertex2):
    m,n=Map.shape
    vertex1=vertex1-1
    vertex2=vertex2-1
    if vertex1 > m or vertex2 > m:
        print("Valid vertex information")
        return None
    Map[vertex1,vertex2]=1
    return Map

setedge(Map,1,2)
setedge(Map,1,4)
setedge(Map,1,5)
setedge(Map,2,3)
setedge(Map,2,4)
setedge(Map,3,4)
setedge(Map,3,8)
setedge(Map,3,9)
setedge(Map,4,5)
setedge(Map,4,6)
setedge(Map,4,7)
setedge(Map,5,6)
setedge(Map,6,7)
setedge(Map,6,12)
setarc(Map,6,13)
setedge(Map,7,11)
setedge(Map,7,12)
setedge(Map,8,9)
setedge(Map,9,10)
setedge(Map,9,11)
setedge(Map,10,11)
setarc(Map,10,13)
setedge(Map,11,12)
setarc(Map,11,13)
setarc(Map,12,13)