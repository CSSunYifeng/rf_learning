import numpy as np
import matplotlib.pyplot as plt

class MultiArmedBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.q_true = np.random.normal(0, 1, num_arms)  # 每个臂真实的平均奖励值
        self.q_estimates = np.zeros(num_arms)  # 每个臂的估计平均奖励值
        self.action_counts = np.zeros(num_arms)  # 每个臂的选择次数

    def select_action(self, epsilon):
        # 使用 ε-greedy 策略选择动作
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.num_arms)  # 随机选择一个臂
        else:
            action = np.argmax(self.q_estimates)  # 选择估计平均奖励值最高的臂
        return action

    def take_action(self, action):
        # 获取选择的臂的奖励
        reward = np.random.normal(self.q_true[action], 1)
        self.action_counts[action] += 1
        self.update_estimates(action, reward)

    def update_estimates(self, action, reward):
        # 使用样本平均法更新臂的估计平均奖励值
        alpha = 1.0 / self.action_counts[action]  # 步长
        self.q_estimates[action] += alpha * (reward - self.q_estimates[action])

# 测试
num_arms = 10
num_steps = 1000
epsilon = 0.1

bandit = MultiArmedBandit(num_arms)

rewards = []
optimal_actions = []
steps = []

for step in range(num_steps):
    action = bandit.select_action(epsilon)
    bandit.take_action(action)
    rewards.append(np.mean(bandit.q_true))  # 记录平均奖励值
    optimal_actions.append(action == np.argmax(bandit.q_true))  # 记录是否选择最优臂
    steps.append(step + 1)  # 记录训练步数

# 绘制训练步数和平均奖励值的变化曲线
plt.plot(steps, rewards)
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Training Performance')
plt.show()
