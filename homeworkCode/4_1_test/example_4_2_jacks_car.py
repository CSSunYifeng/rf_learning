import numpy as np
# import matplotlib.pyplot as plt
from enum import Enum
from scipy.stats import poisson


class Jacks_cars:
    action_enum = [-5,-4,-3,-2,-1,0,1,2,3,4,5] # A到B为正，B到A为负
    reward_space = []
    state_space = []
    car_a_num_max = 20
    car_b_num_max = 20

    action_space = np.zeros((car_a_num_max+1,car_b_num_max+1),dtype=int)
    V_s = np.zeros((car_a_num_max+1,car_b_num_max+1))
    policy_stable = True
    old_action_space = np.zeros((car_a_num_max+1,car_b_num_max+1))
    poisson_properties = dict()
    poisson_up_bound = 11
    lamda_list=[2,3,4]
    
    def __init__(self):
        self.state_space = np.empty((self.car_a_num_max+1,self.car_b_num_max+1),dtype=object)
        for a in range(self.car_a_num_max+1):
            for b in range(self.car_b_num_max+1):
                self.state_space[a,b] = (a,b)
        
        for n in range(self.poisson_up_bound):
            for lam in self.lamda_list:
                key = lam*self.poisson_up_bound + n
                self.poisson_properties[key] = poisson.pmf(n, lam)# self.get_poisson(lam,n) # poisson.pmf(n, lam)
        print(self.poisson_properties)
        

    def poisson_range(self,lam,n):
        key = lam*self.poisson_up_bound+n
        if key not in self.poisson_properties:
            self.poisson_properties[key] = poisson.pmf(n, lam)
        return self.poisson_properties[key]

    def get_p(self,s,r,total_hire,s_after,action):#在s和a条件下，转移到所有s'的概率分布
        value = 0
        a_diff = s_after[0] - s[0] - action
        b_diff = s_after[1] - s[1] + action
        for ha in range(self.poisson_up_bound):
            if ha > total_hire:
                continue
            hb = total_hire - ha
            rea = abs(ha - a_diff)
            if(rea > 0 and hb <= self.poisson_up_bound):
                p_a_hire = self.poisson_range(3,ha) # (4**ha)/math.factorial(ha)*np.exp(-ha)
                p_a_return = self.poisson_range(3,rea) # (3**rea)/math.factorial(rea)*np.exp(-rea)
            else:
                p_a_hire = 0.0
                p_a_return = 0.0
            reb = abs(hb - b_diff)
            if(reb > 0 and hb <= self.poisson_up_bound):
                p_b_hire = self.poisson_range(4,hb) # (4**hb)/math.factorial(hb)*np.exp(-hb)
                p_b_return = self.poisson_range(2,reb) # (3**reb)/math.factorial(reb)*np.exp(-reb)
            else:
                p_b_hire = 0.0
                p_b_return = 0.0
            value += (p_a_hire*p_a_return)*(p_b_hire*p_b_return)
        return value
    
    def get_vs(self,s,gamma):
        v_s = 0
        for a in range(self.car_a_num_max+1):
            for b in range(self.car_b_num_max+1):
                for rd in range(self.poisson_up_bound+self.poisson_up_bound):
                    s_after = (a,b)
                    r = rd*10-2*abs(self.action_space[s])
                    p = self.get_p(s,r,rd,s_after,self.action_space[s])
                    v_s += p*(r+gamma*self.V_s[s_after])
                    # print(p)
        return v_s
    
    def get_v(self):
        delta = 999
        theta = 0.00000000001
        while(delta>theta):
            v = np.copy(self.V_s)
            for a in range(self.car_a_num_max+1):
                for b in range(self.car_b_num_max+1):
                    s = (a,b)
                    print(s)
                    self.V_s[s] = self.get_vs(s,0.9)
            delta =  np.max(np.abs(v - self.V_s))
            print(delta)

    def get_v_test(self):
        delta = 999
        theta = 0.00000000001
        while True:
            v = np.copy(self.V_s)
            for a in range(self.car_a_num_max+1):
                for b in range(self.car_b_num_max+1):
                    s = (a,b)
                    print(s)
                    self.V_s[s] = self.get_vs(s,0.9)
            delta =  np.max(np.abs(v - self.V_s))
            print(delta)
            break

    def get_qs(self,s,action,gamma):
        for a in range(self.car_a_num_max+1):
            for b in range(self.car_b_num_max+1):
                for rd in range(self.car_a_num_max+self.car_b_num_max):
                    s_after = (a,b)
                    r = rd*10
                    v_s += self.get_p(s,r,s_after,action)*(r+gamma*self.V_s[s_after])
        return v_s

    def get_pi(self):
        old_action = np.copy(self.action_space)
        for a in range(self.car_a_num_max+1):
            for b in range(self.car_b_num_max+1):
                max_value = -999
                max_action = old_action[a,b]
                s = (a,b)
                for a in self.action_enum:
                    value = self.get_qs(self,s,a,0.9)
                    if(max_value < value):
                        max_action = a
                        max_value = value
                self.action_space[a,b] = max_action
        print(self.action_space)
        if(old_action != self.action_space):
            self.policy_stable = False
        else:
            self.policy_stable = True
        return self.policy_stable
            
def main():
    jc = Jacks_cars()
    delta = 0.000000001
    idx = 0
    while True:
        print("epoch:{}".format(idx))
        old_values = np.copy(jc.V_s)
        jc.get_v()
        ret = jc.get_pi()
        judge = np.max(np.abs(old_values - jc.V_s))
        if(ret == True):
            if(judge > delta):
                return jc.V_s,jc.action_space
        idx += 1
        print("judge:{}".format(judge))

def main2():
    jc = Jacks_cars()
    delta = 0.000000001
    idx = 0
    while True and idx < 1:
        print("epoch:{}".format(idx))
        old_values = np.copy(jc.V_s)
        jc.get_v_test()
        print(jc.V_s)
        idx += 1
    


if __name__ == "__main__":
    main2()

            
"""
    def state_trans(self,state,action):
        state_after = (0,0)
        if(action == 0):
            return state
        elif(action < 0):
            state_after[1] = state[1]+action
            if(state_after[1])<0:
                action = -state[1]
                state_after[1] = 0
            state_after[0] -= action
            if(state_after[0] > 20):
                state_after[0] = 20
        else:
            state_after[0] = state[0]-action
            if(state_after[0])<0:
                action = state[0]
                state_after[0] = 0
            state_after[1] += action
            if(state_after[1] > 20):
                state_after[1] = 20        
        return state_after    
"""

    
            