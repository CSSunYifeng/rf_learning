import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class Wangge:#状态空间为5*5的矩阵的坐标

    class action_space(Enum):
        Up = 0
        Down = 1
        Left = 2
        Right = 3
    
    class reward_space(Enum):
        board = -1.0
        best = 10.0
        middle = 5.0
        common = 0.0

    def __init__(self):
        self.values = np.zeros((5,5))
        #left
        self.left_benefits = np.zeros((5,5))
        self.left_benefits[:,0] = -1
        self.left_benefits[0,1] = 10
        self.left_benefits[0,3] = 5
        #right
        self.right_benefits = np.zeros((5,5))
        self.right_benefits[:,4] = -1
        self.right_benefits[0,1] = 10
        self.right_benefits[0,3] = 5
        #up
        self.up_benefits = np.zeros((5,5))
        self.up_benefits[0,:] = -1
        self.up_benefits[0,1] = 10
        self.up_benefits[0,3] = 5
        #down
        self.down_benefits = np.zeros((5,5))
        self.down_benefits[4,:] = -1
        self.down_benefits[0,1] = 10
        self.down_benefits[0,3] = 5
        self.benefits = np.stack((self.left_benefits,self.right_benefits,self.up_benefits,self.down_benefits))# R_t
        self.policy = (0.25,0.25,0.25,0.25)# pi_a|s



    def trans_s(self,state,action):
        if state == (0,1):
            return (4,1)
        if state == (0,3):
            return (2,3)
        if action == Wangge.action_space.Left:
            if state[0] == 0:
                return state
            else:
                return (state[0]-1,state[1])
        elif action == Wangge.action_space.Right:
            if state[0] == 4:
                return state
            else:
                return (state[0]+1,state[1])
        elif action == Wangge.action_space.Up:
            if state[1] == 0:
                return state
            else:
                return (state[0],state[1]-1) 
        elif action == Wangge.action_space.Down:
            if state[1] == 4:
                return state
            else:
                return (state[0],state[1]+1) 
            
    def get_p(self,s_after,r,s,a):
        if s == (0,1) and s_after == (4,1):
            g = 1
        if r.value == self.benefits[a.value,s[0],s[1]] and s_after == self.trans_s(s,a):
            return 1
        else:
            return 0
        
    def get_vs(self,s,gamma): # 贝尔曼方程求价值函数
        ret = 0
        for a in Wangge.action_space:
            r_sa = 0
            for i in range(5):#遍历s'
                for j in range(5):
                    for r in Wangge.reward_space:
                        s_after = (i,j)
                        r_sa += self.get_p(s_after,r,s,a)*(self.benefits[a.value,s[0],s[1]]\
                                                         +gamma*self.values[s_after[0],s_after[1]])#self.get_vs(s_after,gamma))
            r_s += self.policy[a.value]*r_sa 
        return r_s
    
    def get_best_vs(self,s,gamma): # 贝尔曼最优状态方程求最优价值函数
        best_r_sa = -99
        for a in Wangge.action_space:
            r_sa = 0
            for i in range(5):
                for j in range(5):
                    for r in Wangge.reward_space:
                        s_after = (i,j)
                        r_sa += self.get_p(s_after,r,s,a)*(self.benefits[a.value,s[0],s[1]]\
                                                         +gamma*self.values[s_after[0],s_after[1]])#self.get_vs(s_after,gamma))
            if r_sa > best_r_sa:
                best_r_sa = r_sa
            r_s = best_r_sa
        return r_s

    def get_v(self): # 参考P73迭代策略评估算法
        delta = 999 #
        v = np.zeros_like(self.values)
        idx = 0
        while(delta > 0.00001):
            v = np.copy(self.values)
            for i in range(5):
                for j in range(5):
                    self.values[i,j] = self.get_vs((i,j),0.9)
            delta =  np.max(np.abs(v - self.values))
            print("epoch:",idx," delta:",delta)
            idx += 1
        print(self.values)


def main():
    c = Wangge()
    c.get_v()

if __name__== "__main__":
    main()

