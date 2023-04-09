import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取身高数据
data = pd.read_csv('height_data.csv')

# 取出身高数据并转化为numpy数组
height = data['height'].values


# 定义高斯分布函数
def gaussian(x, mu, sigma):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)


# 初始化参数
mu1, sigma1 = 170, 5
mu2, sigma2 = 180, 5
pi = 0.5

# 迭代次数
n_iterations = 1000

# EM算法
for i in range(n_iterations):
    # E步：计算后验概率
    gamma1 = pi * gaussian(height, mu1, sigma1)
    gamma2 = (1 - pi) * gaussian(height, mu2, sigma2)
    gamma_sum = gamma1 + gamma2
    gamma1 /= gamma_sum
    gamma2 /= gamma_sum

    # M步：更新参数
    mu1 = np.sum(gamma1 * height) / np.sum(gamma1)
    mu2 = np.sum(gamma2 * height) / np.sum(gamma2)
    sigma1 = np.sqrt(np.sum(gamma1 * (height - mu1) ** 2) / np.sum(gamma1))
    sigma2 = np.sqrt(np.sum(gamma2 * (height - mu2) ** 2) / np.sum(gamma2))
    pi = np.mean(gamma1)


x = np.linspace(150, 195, 5000)
y = pi * gaussian(x, mu1, sigma1) + (1-pi) * gaussian(x, mu2, sigma2)
plt.hist(height, bins=50, density=True, alpha=0.5)
plt.plot(x, y, 'g-', linewidth=2)
plt.show()