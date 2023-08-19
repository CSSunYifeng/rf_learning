import numpy as np
import matplotlib.pyplot as plt

np.random.seed(np.random.randint(0,1000))
class Bandit:
    def __init__(self,mode,stepmode,step,strategy,strategy_value):
        self.q_star = np.random.normal(0,1,10)
        self.dis = False
        self.fix_step = False
        self.A_strategy = 0
        # self.random = np.random.RandomState(seed)
        self.step = 1
        if mode == 'disturbance':
            self.dis = True
        if stepmode == 'constant':
            self.fix_step = True
            self.step = step
        if strategy == 'upper':
            self.A_strategy = strategy_value


    def getRt(self, action_idx:int):
        ret = np.random.normal(self.q_star[action_idx],1)
        return ret
    def getQt(self,Q,R,t):
        if t==0:
            return 0
        else:
            step = 1.0
            if self.fix_step == True:
                step = self.step
            else:
                step = 1.0/t
            return Q+(R-Q)*step
        
    def getBestAction(self):
        return np.argmax(self.q_star)
    
    def getBestPreviewAciont(self,Q):
        return np.argmax(Q)
    
    def getBestPreviewAciont_UpperConfidence(self,Q,Nta,t,c):
        max_value = 5000
        qlen = len(Q)
        cmpQ = np.zeros(qlen)
        for i in range(qlen):
            if Nta[i] == 0:
                cmpQ[i] = Q[i] + max_value
            else:
                cmpQ[i] = Q[i] + c*np.sqrt(np.log(t)/Nta[i])
        return np.argmax(cmpQ)

    def set_qstar_disturbance(self):
        self.q_star += np.random.normal(0,0.01,10)

    def getQlog(self, epsilon_e, total,init_value = 0):
        # 选择初始动作
        Q_log = np.zeros(total)
        A_log = np.zeros(total)
        Q = np.zeros(10)+init_value
        N = np.zeros(10)
        total_R = 0
        best_A_cunt = 0
        for i in range(total):
            c = np.random.random()
            best_action = self.getBestAction()
            A_idx = 0
            if c > epsilon_e:
                if self.A_strategy == False:
                    A_idx = self.getBestPreviewAciont(Q)
                else:
                    A_idx = self.getBestPreviewAciont_UpperConfidence(Q,N,i,2)
            else:
                A_idx = np.random.randint(0,10)
            R = np.zeros(10)
            R[A_idx] = self.getRt(A_idx)
            N[A_idx] = N[A_idx] + 1
            for j in range(10):
                Q[j] = self.getQt(Q[j],R[j],N[j])
            # if i < 1:
            #     Q_log[i] = Q[A_idx]
            # else:
            #     Q_log[i] =Q_log[i-1] + Q[A_idx]
            total_R += R[A_idx]
            Q_log[i] = total_R/(i+1) #Q[A_idx]
            if A_idx == best_action:
                best_A_cunt += 1.0
            A_log[i] = best_A_cunt/(i+1)
            if self.dis:
                self.set_qstar_disturbance()
        return Q_log,A_log

def drawPlot(Q_log,color:str):
    x = np.arange(1, len(Q_log) + 1)
    y = Q_log
    plt.plot(x,y,color)

def main():
    num = 1
    times = 10000
    # 解决 plt 中文显示的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(12,8))
    a1 = fig.add_subplot(2,1,1,label='a1')
    a2 = fig.add_subplot(2,1,2,label='a2')
    a1.set_xlabel('训练步数')
    a1.set_ylabel('平均收益')
    a2.set_xlabel('训练步数')
    a2.set_ylabel('最优动作比例')
    Q = np.zeros(times)
    R = np.zeros(times)
    # paraList = [('withou initial value','common','constant',0.1,0.1,0,'r-'),('with initial value','common','constant',0.1,0,5,'b-')]
    paraList = [('common_epsilon','common','average',None,0.1,0,'common_max',0,'r-'),('upper','common','average',None,0,0,'upper',2,'b-')]
    for experiment_name,mode,stepmode,step,epsilon,init_value,strategy,upper_value,color in paraList:
        print("experiment name:{}".format(experiment_name))
        for i in range(num):
            bandit = Bandit(mode,stepmode,step,strategy,upper_value)
            Q_log_1,R_log_1 = bandit.getQlog(epsilon,times,init_value)
            Q += 1/(i+1)*(Q_log_1 - Q)
            R += 1/(i+1)*(R_log_1 - R)
            print("epoch: ",i)
        a1.plot(Q,color,label=experiment_name)
        a2.plot(R,color,label=experiment_name)
        a1.legend()
        a2.legend()

    plt.savefig('2_6_optimistic_intial_value.png')
    plt.show()
    

def test():
    src = [2,2,2,2,3]
    c = 2
    
    g = [a for a in range(5) if src[a] == 2]
    print(g)

if __name__== "__main__":
    main()