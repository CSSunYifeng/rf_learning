import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self,K):
        self.K = K

        self.values = np.random.randn(K)            # 从标准正态分布采样K个拉杆的收益均值
        self.bestAction = np.argmax(self.values)    # 最优动作索引

        self.QValues = np.zeros(K)  # 价值评估值 
        self.N = np.zeros(K)        # 动作选择次数

        self.setPolicy()            # 缺省策略为greedy

    def setPolicy(self,mode='greedy',epsilnotallow=0):
        self.mode = mode
        self.epsilon = epsilnotallow
        
    def greedy(self):
        return np.random.choice([a for a in range(self.K) if self.QValues[a] == np.max(self.QValues)])

    def epsilonGreedy(self):
        if np.random.binomial(1,self.epsilon) == 1:
            return np.random.randint(self.K)
        else:
            return self.greedy()

    def takeAction(self):
        if self.mode == 'greedy':
            return self.greedy()
        else:
            return self.epsilonGreedy()

    def play(self,times):
        G = 0   # 当前收益
        B = 0   # 当前最优选择次数
        returnCurve = np.zeros(times)       # 收益曲线
        proportionCurve = np.zeros(times)   # 比例曲线
        
        self.QValues = np.zeros(self.K)
        self.N = np.zeros(self.K)

        for i in range(times):
            a = self.takeAction()
            r = np.random.normal(self.values[a],1,1) 
            B += a == self.bestAction

            self.N[a] += 1
            self.QValues[a] += 1/self.N[a]*(r-self.QValues[a])  # 增量式计算均值

            returnCurve[i] = G/(i+1)
            proportionCurve[i] = B/(i+1)
            G += r
    
        return returnCurve,proportionCurve

if __name__ == '__main__':
    K = 10          # 摇臂数
    Num = 100       # 赌博机数量
    times = 2000   # 交互次数
    paraList = [('greedy',0,'greedy'),('epsilon',0.1,'0.1-greedy'),('epsilon',0.01,'0.01-greedy')]

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
    
    # 实例化 Num 个赌博机
    bandits = []
    for i in range(Num):
        bandit = Bandit(K)
        bandit.setPolicy('greedy')
        bandits.append(bandit)

    # 测试三种策略
    for paraMode,paraEpsilon,paraLabel in paraList:
        aveRCurve,avePCurve = np.zeros(times),np.zeros(times)
        for i in range(Num):
            print(paraLabel,i)
            bandits[i].setPolicy(paraMode,paraEpsilon)
            returnCurve,proportionCurve = bandits[i].play(times)
            aveRCurve += 1/(i+1)*(returnCurve-aveRCurve)        # 增量式计算均值
            avePCurve += 1/(i+1)*(proportionCurve-avePCurve)    # 增量式计算均值

        a1.plot(aveRCurve,'-',linewidth=2, label=paraLabel)
        a2.plot(avePCurve,'-',linewidth=2, label=paraLabel)
    
    # a1.legend(fnotallow=10)  # 显示图例，即每条线对应 label 中的内容
    # a2.legend(fnotallow=10)  

    plt.show()
