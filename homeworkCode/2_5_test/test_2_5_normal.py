import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(555)
class Bandit:
    def __init__(self):
        self.q_star = np.random.normal(0,1,10)

    def getRt(self, action_idx:int):
        ret = np.random.normal(self.q_star[action_idx],1)
        return ret
    def getQt(self,Q,R,t):
        if t==0:
            return 0
        else:
            return Q+(R-Q)/t

    def getQlog(self, epsilon_e, total):
        # 选择初始动作
        Q_log = np.zeros(total)
        Q = np.zeros(10)
        N = np.zeros(10)
        for i in range(total):
            c = np.random.rand()
            A_idx = 0
            if c > epsilon_e:
                A_idx = np.argmax(Q)
            else:
                A_idx = np.random.randint(0,10)
            R = np.zeros(10)
            R[A_idx] = R[A_idx] + self.getRt(A_idx)
            N[A_idx] = N[A_idx] + 1
            for j in range(10):
                Q[j] = self.getQt(Q[j],R[j],N[j])
            # if i < 1:
            #     Q_log[i] = Q[A_idx]
            # else:
            #     Q_log[i] =Q_log[i-1] + Q[A_idx]
            Q_log[i] = Q[A_idx]
        return Q_log

def drawPlot(Q_log,color:str):
    x = np.arange(1, len(Q_log) + 1)
    y = Q_log
    plt.plot(x,y,color)
    

def main():
    epsilon_e_1 = 0
    epsilon_e_2 = 0.01
    epsilon_e_3 = 0.1
    num = 1000
    times = 2000
    Q_1,Q_2,Q_3 = np.zeros(times),np.zeros(times),np.zeros(times)
    for i in range(num):
        bandit = Bandit()
        Q_log_1,Q_log_2,Q_log_3 = bandit.getQlog(epsilon_e_1,times),bandit.getQlog(epsilon_e_2,times),bandit.getQlog(epsilon_e_3,times)
        Q_1 += 1/(i+1)*(Q_log_1 - Q_1)
        Q_2 += 1/(i+1)*(Q_log_2 - Q_2)
        Q_3 += 1/(i+1)*(Q_log_3 - Q_3)
        print("round: ",i)
    drawPlot(Q_1,'b')
    drawPlot(Q_2,'r')
    drawPlot(Q_3,'y')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('result')
    plt.show()

def test():
    src = [2,2,2,2,3]
    c = 2
    
    g = [a for a in range(5) if src[a] == 2]
    print(g)

if __name__== "__main__":
    main()