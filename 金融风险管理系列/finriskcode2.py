import numpy as np
from scipy.optimize import minimize
import statsmodels.api as sm

# 生成1000个在-0.05到0.15之间的模拟收益率
Rp = (np.random.rand(1000, 1) - 0.5) / 5 + 0.05
Rf = 0.01
# 生成1000个在-0.05到0.05间的模拟收益率
Rm = (np.random.rand(1000, 1) - 0.5) / 10

# CAPM模型
Rex = Rp - Rf
# 增加常数项
X = sm.add_constant(Rm - Rf)
res = sm.OLS(Rex, X).fit()
# 回归结果
print(res.summary())

alpha = res.params[0]
beta = res.params[1]
print('alpha:', alpha)
print('beta:', beta)

# 特雷诺比率
TR = (np.mean(Rp) - Rf) / beta
print('特雷诺比率', TR)

# 詹森alpha
Jalpha = np.mean(Rp) - Rf - beta * (np.mean(Rm - Rf))
print('詹森alpha', Jalpha)

#APT
# 生成10个股票期望收益率
Rp = (np.random.rand(10, 1) - 0.5)  + 0.05
Rf = 0.01
# 生成3个风险收益率
RFactor = (np.random.rand(10, 3) - 0.5) / 10
Rex = Rp - Rf
res = sm.OLS(Rex, RFactor-Rf).fit()
# 回归结果
print(res.summary())
#APT收益率
APTR=np.sum((RFactor-Rf)*res.params,axis=1)
print('APT收益率', APTR)
#套利空间
print('套利空间', APTR-Rex.T)