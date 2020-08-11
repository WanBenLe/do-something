import numpy as np
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 生成服从标准正态分布的收益率数据
Rp = np.random.randn(1000, 1)
Rf = 0.01
Rm = np.random.rand(1000, 1)

# 计算标准正态分布0和极大值的CDF
print('0的CDF', st.norm.cdf(0))
print('999999的CDF', st.norm.cdf(999999))
# 计算PDF
pdf_rp = st.norm.pdf(Rp)
plt.title('PDF')
plt.hist(pdf_rp)
plt.show()

# 计算标准正态分布0.5和1的ppf
print('0.5的ppf', st.norm.ppf(0.5))
print('1的ppf', st.norm.ppf(1))
# 均值
print('均值', np.mean(Rp))
# 分位数
print('5分位数', np.percentile(Rp, 5))
# 中位数
print('50分位数', np.percentile(Rp, 50))
print('中位数', np.median(Rp))
# 方差
print('方差', np.var(Rp))
# 标准差
print('标准差', np.std(Rp))
# 偏度
print('偏度', st.skew(Rp)[0])
# 峰度
print('峰度', st.kurtosis(Rp)[0])
# 矩
print('2阶矩', st.moment(Rp, 2)[0])
print('协方差', np.cov(Rp.T, Rm.T)[0, 1])
print('皮尔逊相关系数', st.pearsonr(Rp.reshape(-1), Rm.reshape(-1))[0])
print('斯皮尔曼相关系数', st.pearsonr(Rp.reshape(-1), Rm.reshape(-1))[0])
# 随机变量线性变换
Rp_m = np.mean(Rp)
Rm_m = np.mean(Rm)
Rp_v = np.var(Rp)
Rm_v = np.var(Rm)

a = 0.02
b = 1.5
print('均值1:', np.mean(a + b * Rp))
print('均值2:', a + np.mean(b * Rp))
print('方差1:', np.var(a + b * Rp))
print('方差2:', b ** 2 * Rp_v)
print('标准差1:', np.std(a + b * Rp))
print('标准差2:', b * np.std(Rp))

print('均值和1:', np.mean(Rp + Rm))
print('均值和2:', Rp_m + Rm_m)
print('方差和1:', np.var(Rp + Rm))
print('方差和2:', Rp_v + Rm_v + 2 * np.cov(Rp.T, Rm.T)[0, 1])

# 生成5个投资组合收益率
Rp = (np.random.rand(1000, 5) - 0.501) / 5
w = np.array([0.1, 0.11, 0.25, 0.35, 0.19])
print('总投资组合收益率', np.sum(Rp @ w))
print('投资组合方差1', np.var(Rp @ w))
print('投资组合方差2', w @ np.cov(Rp.T) @ w.T)
print('投资组合波动率', (w @ np.cov(Rp.T) @ w.T) ** 0.5)
