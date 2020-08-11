# 测试库安装
try:
    import numpy as np
    import statsmodels
    import pandas
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize

    print('安装成功!')
except:
    print('有库安装错误')

# 模拟收益率并画图
data = (np.random.rand(1000, 1) - 0.5) / 5  # 生成1000个在-0.1到0.1之间的小数模拟收益率
plt.hist(data)  # 画出直方图
plt.show()
print('平均收益率', np.mean(data))  # 期望
print('波动率', np.std(data))  # 波动率
print('5%VaR', np.percentile(data, 5))  # 5%下分位数

# 绝对风险与相对风险
P = 1000000
Rp = (np.random.rand(1000, 1) - 0.5) / 5  # 生成1000个在-0.1到0.1之间的小数模拟收益率
print('绝对风险:', np.std(Rp) * P)

P = 1000000
Rp = (np.random.rand(1000, 1) - 0.5) / 5  # 生成1000个在-0.1到0.1之间的小数模拟收益率
Rb = (np.random.rand(1000, 1) - 0.5) / 10  # 生成1000个在-0.05到0.05之间的小数模拟收益率
print('TEV:', np.std(Rp - Rb))
print('相对风险:', np.std(Rp - Rb) * P)

# SR与IR
# 生成1000个在-0.05-0.15间的模拟收益率
Rp = (np.random.rand(1000, 1) - 0.5) / 5 + 0.05
# 生成1000个在-0.01到0.01之间的模拟收益率
Rf = (np.random.rand(1000, 1) - 0.5) / 50
print('夏普比率:', np.mean(Rp - Rf) / np.std(Rp))
# 生成1000个在-0.05-0.15间的模拟收益率
Rp = (np.random.rand(1000, 1) - 0.5) / 5 + 0.05
# 生成1000个在-0.05到0.05之间的模拟收益率
Rb = (np.random.rand(1000, 1) - 0.5) / 10
TEV = np.std(Rp - Rb)
print('信息比率:', np.mean(Rp - Rb) / TEV)

# 信息比率的优缺点
# 生成1000个在-0.15到0.05间的模拟收益率
Rp = (np.random.rand(1000, 1) - 0.5) / 5 - 0.05
# 生成1000个在-0.2到0之间的模拟收益率
Rb = (np.random.rand(1000, 1) - 0.5) / 5 - 0.1
TEV = np.std(Rp - Rb)
print('信息比率1:', np.mean(Rp - Rb) / TEV)
# 生成1000个在-0.1到0.00之间的模拟收益率
Rb = (np.random.rand(1000, 1) - 0.5) / 10 - 0.05
Rp = Rb + (np.random.rand(1000, 1)) / 100
TEV = np.std(Rp - Rb)
print('信息比率2:', np.mean(Rp - Rb) / TEV)

# TEV的两种计算方式
# 生成1000个在-0.05-0.15间的模拟收益率
Rp = (np.random.rand(1000, 1) - 0.5) / 5 + 0.05
# 生成1000个在-0.05到0.05之间的模拟收益率
Rb = (np.random.rand(1000, 1) - 0.5) / 10
TEV = np.std(Rp - Rb)
sd_b = np.std(Rb)
sd_p = np.std(Rp)
cov = np.cov(Rb.T, Rp.T)[0, 1]  # 计算协方差
print('TEV1:', TEV)
TEV2 = (sd_b ** 2 + sd_p ** 2 + 2 * cov) ** 0.5
print('TEV2:', TEV2)

# 投资组合优化: 分析与设置
n = 10
R = (np.random.rand(1000, n) - 0.5) / 5 * np.random.rand(1, 10) * 5
print(np.mean(R, 0))
x0 = [1 / n] * n  # 初始化参数——等权
# 权重之和为1的约束
cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
# 各个权重应该大于等于0的约束
bnds = eval('(0, None),' * n)


# 最大夏普比率优化
# 最小化负的夏普比率
def maxsr(w, var):
    w = np.asarray(w)
    r = np.sum(w * var)
    sr = -r / np.std(w * var)
    return sr


res = minimize(maxsr, x0, args=R, method='SLSQP', tol=1e-6, constraints=cons, bounds=bnds)
weight = res.x
print('资产权重', weight)
print('资产权重和', np.sum(weight))
print('最优夏普比率:', -res.fun)


# 最小波动率
# 最小化负的波动率
def maxsr(w, var):
    w = np.asarray(w)
    sd = np.std(w * var)
    return -sd


res = minimize(maxsr, x0, args=R, method='SLSQP', tol=1e-6, constraints=cons, bounds=bnds)
weight = res.x
print('资产权重', weight)
print('资产权重和', np.sum(weight))
print('最优波动率:', -res.fun)
