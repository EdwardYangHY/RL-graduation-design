# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 11:22:22 2020

@author: Edward rewriting from Qiang Ye

License:MIT
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


#Grid是所有之后需要用到的类的基础
#这个基础类是在定义每一个格子的值，即，在之后的大网格中，每一个小格子拥有的元组
class Grid(object):
    def __init__(self,x:int=None,
                      y:int=None,
                      dtype:int=0,           #源代码此处是type
                      reward:float=0.0,
                      value:float=0.0):
        self.x=x
        self.y=y
        self.dtype=dtype                     #源代码是self.type = value，不解
        self.reward=reward
        self.value=value
        self.name=None
        self._update_name()
        
    #给出name：即尺寸？
    def _update_name(self):
        self.name="X{0}-Y{1}".format(self.x,self.y)
        
    def __str__(self):
         #源代码还有reward，但是实际上没用到
        return "name:{4},x:{0},y:{1},type:{2},value{3}".format(self.x,
                                                               self.y,
                                                               self.dtype,   
                                                               self.value,
                                                               self.name)

#这个类可以理解为Grid类的集成，即一个由grid类组成的矩阵    
class GridMatirx(object):
    #格子矩阵，通过不同的设置模拟不同的格子世界环境
    def __init__(self,n_width:int,
                      n_height:int,
                      default_type:int=0,
                      default_reward:float=0.0,
                      default_value:float=0.0):
        self.grid=None
        self.n_height=n_height
        self.n_width=n_width
        self.len=n_width*n_height
        self.default_type=default_type
        self.default_reward=default_reward
        self.default_value=default_value
        self.reset()
        
    def reset(self):
        self.grids=[]
        for x in range(self.n_height):
            for y in range(self.n_width):
                self.grids.append(Grid(x,
                                       y,
                                       self.default_type,
                                       self.default_reward,
                                       self.default_value))
                
    def get_grid(self,x,y=None):
        '''
        获得单个格子（grid）的信息
        输入可以是一x和y索引，也可以是x的一个单独元组
        所以在输入时要进行判断
        '''
        xx,yy=None,None
        if isinstance(x, int):
            xx,yy=x,y
        elif isinstance(x, tuple):
            xx,yy=x[0],x[1]
        assert(xx>=0 and yy>=0 and xx<self.n_width and yy<self.n_height),\
            "coordinates should be in reasonable range"
        index=yy*self.n_width+xx              #看得出来，网格世界是一维离散的世界
        return self.grids[index]
    
    def set_reward(self,x,y,reward):
        grid=self.get_grid(x,y)
        if grid is not None:
            grid.reward=reward
        else:
            raise("grid doesn't exist")
        
    def set_value(self,x,y,value):
        grid=self.get_grid(x,y)
        if grid is not None:
            grid.value=value
        else:
            raise("grid doesn't exist")    
    
    def set_type(self,x,y,dtype):
        grid=self.get_grid(x,y)
        if grid is not None:
            grid.dtype=dtype
        else:
            raise("grid doesn't exist")   
        
    def get_reward(self,x,y):
        grid=self.get_grid(x,y)
        if grid is None:
            return None
        else:
            return grid.reward
        
    def get_value(self,x,y):
        grid=self.get_grid(x,y)
        if grid is None:
            return None
        else:
            return grid.value
        
    def get_type(self,x,y):
        grid=self.get_grid(x,y)
        if grid is None:
            return None
        else:
            return grid.dtype
        
#通过上述的两个基础类来构造环境
class GridWorldEnv(gym.Env):
    '''
    继承gym.Env类，正式开始环境配置
    '''
    metadata={
        'render.modes':['human','rgb_array'],
        'video.frames_per_second':30
        }
    
    def __init__(self,n_width:int=10,
                      n_height:int=7,
                      u_size=40,
                      default_reward:float=0,
                      default_type=0,
                      windy=False):       #windy是啥啊，有风模型吗？
        self.u_size=u_size                #每个格子的像素大小
        self.n_width=n_width
        self.n_height=n_height
        self.width=u_size*n_width         #像素大小*数量=实际占用大小
        self.height=u_size*n_height
        self.default_reward=default_reward
        self.default_type=default_type
        self._adjust_size()
        
        self.grids=GridMatirx(n_width=self.n_width,
                              n_height=self.n_height,
                              default_reward=self.default_reward,
                              default_type=self.default_type,
                              default_value=0.0)
        
        self.reward=0
        self.action=None
        self.windy=windy      #是否是个有风环境
        
        #动作空间，对于格子网络来说主要是看有没有停止这个动作
        #如果没有停止，则是4；有停止则是5
        self.action_space=spaces.Discrete(4)
        self.observation_space=spaces.Discrete(self.n_width*self.n_height)
        
        #空间的坐标原点是左下角，和pyglet是一致的
        #设置起点终点和特殊奖励的格子以及障碍就可以实现不同的环境
        
        self.ends=[(7,3)]   #终止点可以有很多
        self.start=(0,3)    #开始点只能一个（单agent系统）
        self.types=[]        #通过传入点(3,2,1)将(3,2)的type设置为1
        self.rewards=[]      #通过传入点(3,3,3)将(3,3)的奖励设置为3
        self.refresh_setting()
        self.viewer=None    #图形接口
        self.seed()        #产生随机种子
        self.reset()       #将start点转为s格式,这里面有设置self.state
        
    def _adjust_size(self):
        #主要用来调整尺寸大小，最大不超过800*8000
        pass
    
    def seed(self,seed=None):
        #产生一个随机化需要的中子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random,seed=seeding.np_random(seed)
        return [seed]
    
    def step(self,action):
        assert self.action_space.contains(action),\
            "%r (%s) invalid" % (action,type(action))
        self.action=action
        old_x,old_y=self._state_to_xy(self.state)      #state在哪定义的呢？？？
        new_x,new_y=old_x,old_y
        
        #如果是有风环境的设置（有风环境请自行百度）
        if self.windy:
            if new_x in [3,4,5,8]:
                new_y+=1
            elif new_x in [6,7]:
                new_y+=2
        
        #0：左  1：右  2：上  3：下
        if action==0: new_x-=1
        elif action==1: new_x+=1
        elif action==2: new_y+=1
        elif action==3: new_y-=1
        elif action==4: new_x,new_y=new_x,new_y        #hold
        
        elif action==5: new_x,new_y=new_x-1,new_y-1    #左下角
        elif action==6: new_x,new_y=new_x+1,new_y-1    #右下角
        elif action==7: new_x,new_y=new_x-1,new_y+1    #左上角
        elif action==8: new_x,new_y=new_x+1,new_y+1    #右上角
        
        #边界效应：走到边边了会怎样？会不动撒
        if new_x < 0: new_x=0
        if new_x >= self.n_width: new_x=self.n_width-1
        if new_y < 0: new_y=0
        if new_y >= self.n_height: new_y=self.n_height-1
        
        if self.grids.get_type(new_x, new_y)==1:
            new_x,new_y=old_x,old_y
            
        self.reward=self.grids.get_reward(new_x, new_y)
        done=self._is_end_state(new_x,new_y)
        self.state=self._xy_to_state(new_x,new_y)
        info={"x":new_x,"y":new_y,"grids":self.grids}
        return self.state,self.reward,done,info
    
    def _state_to_xy(self,s):
        x=s % self.n_width
        y=int((s-x)/self.n_width)
        return x,y
    
    def _xy_to_state(self,x,y=None):
        if isinstance(x,int):
            assert(isinstance(y, int)),"Incomplete Position info"
            return x+self.n_width*y
        elif isinstance(x, tuple):
            return x[0]+self.n_width*x[1]
        return -1       #输入错误导致的未知状态
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    def refresh_setting(self):
        '''
        用户创捷格子类后可能会修改其中某些格子的类型或者奖励设置，
        修改设置后通过调用方法使得设置生效
        '''
        for x,y,r in self.rewards:
            self.grids.set_reward(x, y,r)
        for x,y,t in self.types:
            self.grids.set_type(x, y, t)
    
    def reset(self):
        self.state=self._xy_to_state(self.start)
        return self.state
    
    def _is_end_state(self,x,y=None):
        #出了点小问题
        if y is not None:
            xx,yy=x,y
        elif isinstance(x,int):
            xx,yy=self._state_to_xy(x)
        else:
            assert(isinstance(x,tuple)),"Incomplete coordinate values"
            xx,yy=x[0],x[1]
        for end in self.ends:
            if xx==end[0] and yy==end[1]:
                return True
        return False
    
    #图形化界面。虽然图形化并不是必要的，但是其作为展示工具还是有很大的价值
    #图形化可以让人更加直观的了解到 模型的训练过程
    def render(self,mode='human',close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer=None
            return
        zero=(0,0)
        u_size=self.u_size
        m=2                  #格子之间的间隙
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width,self.height)
            
            # 在Viewer里绘制一个几何图像的步骤如下：
            # the following steps just tells how to render an shape in the environment.
            # 1. 建立该对象需要的数据本身
            # 2. 使用rendering提供的方法返回一个geom对象
            # 3. 对geom对象进行一些对象颜色、线宽、线型、变换属性的设置（有些对象提供一些个
            #    性化的方法来设置属性，具体请参考继承自这些Geom的对象），这其中有一个重要的
            #    属性就是变换属性，
            #    该属性负责对对象在屏幕中的位置、渲染、缩放进行渲染。如果某对象
            #    在呈现时可能发生上述变化，则应建立关于该对象的变换属性。该属性是一个
            #    Transform对象，而一个Transform对象，包括translate、rotate和scale
            #    三个属性，每个属性都由以np.array对象描述的矩阵决定。
            # 4. 将新建立的geom对象添加至viewer的绘制对象列表里，如果在屏幕上只出现一次，
            #    将其加入到add_onegeom(）列表中，如果需要多次渲染，则将其加入add_geom()
            # 5. 在渲染整个viewer之前，对有需要的geom的参数进行修改，修改主要基于该对象
            #    的Transform对象
            # 6. 调用Viewer的render()方法进行绘制
            
            '''
            该段代码设置了格子之间的间隙m，可以不用
            for i in range(self.n_width+1):
                line=rendering.Line(start=(i*u_size,0),
                                    end=(i*u_size,u_size*self.n_height))
                line.set_color(0.5, 0, 0)
                self.viewer.add_geom(line)
            for i in range(self.n_height):
                line=rendering.Line(start=(0,i*u_size),
                                    end=(u_size*self.n_width,i*u_size))
                line.set_color(0, 0, 1)
                self.viewer.add_geom(line)
            '''
            
            #绘制格子
            for x in range(self.n_width):
                for y in range(self.n_height):
                    v=[(x*u_size+m,y*u_size+m),
                       ((x+1)*u_size-m,y*u_size+m),
                       ((x+1)*u_size-m,(y+1)*u_size-m),
                       (x*u_size+m,(y+1)*u_size-m)]
                    
                    rect=rendering.FilledPolygon(v)
                    r=self.grids.get_reward(x, y)/10
                    if r<0:
                        #以下r==-1和r==1都是为2020年数模B第二问，第四关设计
                        #如果不需要可以删掉，保留else下即可
                        #r==-10/10==-1是表示矿山
                        '''
                        if r==-1:
                            rect.set_color(160/255,82/255,45/255)
                        else:
                            rect.set_color(0.9-r, 0.9+r, 0.9+r)
                        '''
                        rect.set_color(0.9-r, 0.9+r, 0.9+r)
                    elif r>0:
                        #r==10/10==1是表示村庄
                        '''
                        if r==1:
                            rect.set_color(64/255,224/255,205/255)
                        else:
                            rect.set_color(0.3, 0.5+r, 0.3)
                        '''
                        rect.set_color(0.3, 0.5+r, 0.3)
                    else:
                        rect.set_color(0.9, 0.9, 0.9)
                    self.viewer.add_geom(rect)
                    #边框绘制
                    v_outline=[(x*u_size+m,y*u_size+m),
                               ((x+1)*u_size-m,y*u_size+m),
                               ((x+1)*u_size-m,(y+1)*u_size-m),
                               (x*u_size+m,(y+1)*u_size-m)]
                    outline=rendering.make_polygon(v_outline,False)
                    outline.set_linewidth(3)
                    
                    if self._is_end_state(x,y):
                        outline.set_color(0.9, 0.9, 0)
                        self.viewer.add_geom(outline)
                    if self.start[0]==x and self.start[1]==y:
                        outline.set_color(0.5, 0.5, 0.8)
                        self.viewer.add_geom(outline)
                    if self.grids.get_type(x, y)==1:
                        rect.set_color(0.5, 0.5, 0.5)
                    else:
                        pass
            self.agent=rendering.make_circle(u_size/4,30,True)
            self.agent.set_color(1.0, 1.0, 0.0)
            self.viewer.add_geom(self.agent)
            self.agent_trans=rendering.Transform()
            self.agent.add_attr(self.agent_trans)
            
        #更新agent位置
        x,y=self._state_to_xy(self.state)
        self.agent_trans.set_translation((x+0.5)*u_size, (y+0.5)*u_size)
            
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    
    def SkullAndTreasure(self):
        '''骷髅与钱币示例，解释随机策略的有效性 David Silver 强化学习公开课第六讲 策略梯度
        Examples of Skull and Money explained the necessity and effectiveness
        '''
        env = GridWorldEnv(n_width=5,
                           n_height = 2,
                           u_size = 60,
                           default_reward = -1,
                           default_type = 0,
                           windy=False)
        env.action_space = spaces.Discrete(4) # left or right
        env.start = (0,1)
        env.ends = [(2,0)]
        env.rewards=[(0,0,-100),(2,0,100),(4,0,-100)]
        env.types = [(1,0,1),(3,0,1)]
        env.refresh_setting()
        return env 
    
    def SimpleGridWorld(self):
        env=GridWorldEnv(n_width=10,
                         n_height=7,
                         u_size=60,
                         default_reward=-1,
                         default_type=0,
                         windy=False)
        env.start=(0,3)
        env.ends=[(7,3)]
        env.rewards=[(7,3,1),(4,5,-0.5)]
        env.types=[(4,3,1),(4,2,1),(4,4,1)]
        env.refresh_setting()
        return env
    
    def DessertCrossing(self):
        #毕设模型
        #2020年数模B第二问，第四关
        env=GridWorldEnv(n_width=5,
                         n_height=5,
                         u_size=60,
                         default_reward=0,
                         default_type=0,
                         windy=False)
        env.action_space=spaces.Discrete(5)
        env.start=(0,4)
        env.ends=[(4,0)]
        #reward设置10和-10都是特殊的颜色
        env.rewards=[(2,1,1),(3,2,-1),(4,0,2)]
        env.refresh_setting()
        return env
    
    def DessertCrossing_simple(self):
        #毕设模型
        #2020年数模B第二问，第四关
        #为了收敛速度，做一个合理的改良
        env=GridWorldEnv(n_width=5,
                         n_height=5,
                         u_size=60,
                         default_reward=0,
                         default_type=0,
                         windy=False)
        env.action_space=spaces.Discrete(5)
        env.start=(0,4)
        env.ends=[(4,0)]
        #reward设置10和-10都是特殊的颜色
        env.rewards=[(2,1,1),(3,2,-1),(4,0,2)]
        #给不必要的路设置障碍
        env.types=[(0,0,1),(1,0,1),(2,0,1),
                   (0,1,1),(1,1,1),
                   (4,2,1),
                   (3,3,1),(4,3,1),
                   (3,4,1),(4,4,1)]
        env.refresh_setting()
        return env
   
    def WindyGridWorld(self):
        env=GridWorldEnv(n_width=10,
                         n_height=7,
                         u_size=60,
                         default_reward=-1,
                         default_type=0,
                         windy=True)
        env.start=(0,3)
        env.ends=[(7,3)]
        env.rewards=[(7,3,1)]
        env.refresh_setting()
        return env

#,(0,2,1),(1,4,1),(2,1,1),(2,4,1),(3,1,1),(3,2,1),(3,3,1),(3,4,1),(4,1,1),(4,2,1),(4,3,1),(4,4,1)

class MA_GridWorldEnv(gym.Env):
    '''
    继承gym.Env类，正式开始环境配置
    多智能体类别
    一个专用于警察抓小偷的类别
    '''
    metadata={
        'render.modes':['human','rgb_array'],
        'video.frames_per_second':30
        }
    
    def __init__(self,n_width:int=10,
                      n_height:int=7,
                      u_size=40,
                      default_reward:float=0,
                      default_type=0,
                      windy=False):       #windy是啥啊，有风模型吗？
        self.u_size=u_size                #每个格子的像素大小
        self.n_width=n_width
        self.n_height=n_height
        self.width=u_size*n_width         #像素大小*数量=实际占用大小
        self.height=u_size*n_height
        self.default_reward=default_reward
        self.default_type=default_type
        self.agent1=None
        self.agent2=None
        self.agent3=None
        
        self.grids=GridMatirx(n_width=self.n_width,
                              n_height=self.n_height,
                              default_reward=self.default_reward,
                              default_type=self.default_type,
                              default_value=0.0)
        
        self.viewer=None    #图形接口
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            
    def set_pos(self,list_of_pos):
        #固定一下输入
        #输入格式为一个list, 有六个数字
        self.x_1=list_of_pos[0]
        self.y_1=list_of_pos[1]
        self.x_2=list_of_pos[2]
        self.y_2=list_of_pos[3]
        self.x_3=list_of_pos[4]
        self.y_3=list_of_pos[5]
        
    #图形化界面。虽然图形化并不是必要的，但是其作为展示工具还是有很大的价值
    #图形化可以让人更加直观的了解到 模型的训练过程
    def render(self,mode='human',close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer=None
            return
        u_size=self.u_size
        m=2                  #格子之间的间隙
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width,self.height)
            
            #绘制格子
            for x in range(self.n_width):
                for y in range(self.n_height):
                    v=[(x*u_size+m,y*u_size+m),
                       ((x+1)*u_size-m,y*u_size+m),
                       ((x+1)*u_size-m,(y+1)*u_size-m),
                       (x*u_size+m,(y+1)*u_size-m)]
                    
                    rect=rendering.FilledPolygon(v)
                    rect.set_color(0.9, 0.9, 0.9)
                    self.viewer.add_geom(rect)
                    #边框绘制
                    v_outline=[(x*u_size+m,y*u_size+m),
                               ((x+1)*u_size-m,y*u_size+m),
                               ((x+1)*u_size-m,(y+1)*u_size-m),
                               (x*u_size+m,(y+1)*u_size-m)]
                    outline=rendering.make_polygon(v_outline,False)
                    outline.set_linewidth(3)
                    
            self.agent1=rendering.make_circle(u_size/4,30,True)
            self.agent1.set_color(1.0, 1.0, 0.0)
            self.viewer.add_geom(self.agent1)
            self.agent1_trans=rendering.Transform()
            self.agent1.add_attr(self.agent1_trans)
            
            self.agent2=rendering.make_circle(u_size/4,30,True)
            self.agent2.set_color(1.0, 1.0, 0.0)
            self.viewer.add_geom(self.agent2)
            self.agent2_trans=rendering.Transform()
            self.agent2.add_attr(self.agent2_trans)
            
            self.agent3=rendering.make_circle(u_size/4,30,True)
            self.agent3.set_color(0.0, 1.0, 1.0)
            self.viewer.add_geom(self.agent3)
            self.agent3_trans=rendering.Transform()
            self.agent3.add_attr(self.agent3_trans)
            
        #更新agent1位置
        self.agent1_trans.set_translation((self.x_1+0.5)*u_size, (self.y_1+0.5)*u_size)
        self.agent2_trans.set_translation((self.x_2+0.5)*u_size, (self.y_2+0.5)*u_size)
        self.agent3_trans.set_translation((self.x_3+0.5)*u_size, (self.y_3+0.5)*u_size)
            
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    
    def ChasingAndBlocking(self):
        env=MA_GridWorldEnv(n_width=5,
                         n_height=5,
                         u_size=60,)
        return env














































