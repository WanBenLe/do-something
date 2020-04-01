import pandas as pd
from sklearn.linear_model import Lasso
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import warnings
from sklearn.svm import SVR

'''
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)
'''
warnings.filterwarnings("ignore")
# 读取数据.分割
data0 = pd.read_csv('avg_amdata.csv')

data0 = data0[data0['codate'] > 200400]
data1 = data0[data0['codate'] <= 201400]
return_1 = data1['retx'].values
data1.index = range(len(data1['codate']))
bt_1 = data1[['codate', 'retx']]
bt_1.index = range(len(bt_1['codate']))
data1 = data1.drop(columns=['retx'])
data7 = data1.copy(deep=True)

Sdata0= pd.read_csv('avg_amdataS.csv')
Sdata0 = Sdata0[Sdata0['codate'] > 200400]
Sdata1 = Sdata0[Sdata0['codate'] <= 201400]
Sdata1.index = range(len(Sdata1['codate']))
Sdata1 = Sdata1.drop(columns=['retx'])
Sreturn_1 = Sdata1['retx'].values
Sdata7 = Sdata1.copy(deep=True)
data5 = Sdata1.ix[:, 2:].values
data5[np.isnan(data5)] = 0.0


data2 = data0[data0['codate'] > 201400]
data2.index = range(len(data2['codate']))
return_2 = data2['retx'].values
bt_16 = data2[['codate', 'retx']]
bt_16.index = range(len(bt_16['codate']))
data2 = data2.drop(columns=['retx'])



# 缺失值处理
return_2[np.isnan(return_2)] = 0.0
data2[np.isnan(data2)] = 0.0

data6 = data2.ix[:, 2:].values
data6[np.isnan(data6)] = 0.0
data6 = data6

# 相关系数
print(np.corrcoef(data6))

# 传统Lasso
'''
alpha_ridge = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

coeffs = {}
for alpha in alpha_ridge:
    r = Lasso(alpha=alpha, normalize=True, max_iter=1000000)
    r = r.fit(data1, return_1)

grid_search = GridSearchCV(Lasso(alpha=alpha, normalize=True), scoring='neg_mean_squared_error',
                           param_grid={'alpha': alpha_ridge}, cv=10, n_jobs=-1)
grid_search.fit(data1, return_1)
alpha = alpha_ridge
rmse = list(np.sqrt(-grid_search.cv_results_['mean_test_score']))

plt.figure(figsize=(6, 5))
lasso_cv = pd.Series(rmse, index=alpha)
lasso_cv.plot(title="Validation - LASSO", logx=True)
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()

lasso = Lasso(alpha=1e-6, normalize=True, max_iter=1e6)
lasso = lasso.fit(data1, return_1)
coef = pd.Series(lasso.coef_)
print(coef)
print("Lasso选中了" + str(sum(coef != 0)) + "个变量，并移除了其他" + str(sum(coef == 0)) + "个变量")
coff_d = pd.DataFrame(data=coef, columns=['Lasso-coff'])
coff_d.to_csv('Lasso1_coff.csv', index=False, header=False)

data3 = pd.read_csv('Lasso1_coff.csv').values
index_0 = list(np.where(np.abs(data3) != 0.0)[0])
data4 = data2.ix[:, index_0]

x = sm.add_constant(data4)
y = return_2
regr = sm.OLS(y, x)
res = regr.fit()
print(res.summary())
r_x = np.dot(res.params[1:].T, data4.values.T)
r_x += res.params[0]
bt_1['retx_f'] = pd.DataFrame(data=np.array([r_x]).T, columns=['a'])['a']

date = list(set(bt_1['codate']))
date.sort()
for i in range(len(date)):
    temp = bt_1[bt_1['codate'] == date[i]]
    temp.sort_values("retx_f", inplace=True, ascending=False)
    temp.index = range(len(temp['retx_f']))
    temp = temp[0:10]
    a = np.mean(temp['retx_f'])
    b = np.mean(temp['retx'])
    temp_d = np.array([a, b])
    if i == 0:
        bt_data = temp_d
    else:
        bt_data = np.vstack((bt_data, temp_d))
print(bt_data)

renturn_x = [1.0]
for i in range(len(date) - 1):
    if bt_data[i][0] >= 0:
        renturn_x.append(renturn_x[-1] * (1 + bt_data[i + 1][1]))
    else:
        renturn_x.append(renturn_x[-1])
print(renturn_x)

plt.plot(range(len(renturn_x)), renturn_x, label='Frist line', linewidth=3, color='r', marker='o',
         markerfacecolor='blue', markersize=12)
plt.xlabel('period')
plt.ylabel('value')
plt.title('Lasso Backtest')
plt.legend()
plt.show()
'''

# 协方差计算
data5 = pd.DataFrame(data=data5)

data5['retx'] = pd.DataFrame(data=np.array([Sreturn_1]).T, columns=['a'])['a']
data5['codate'] = Sdata1['codate']
date_x = list(set(data5['codate'].values))
date_x.sort()
for i in range(len(date_x)):
    a = data5[data5['codate'] == date_x[i]]
    a.index = range(len(a['retx']))
    c = np.array([np.nanmean(a['retx'])])
    a = a.ix[:, :-1].values
    a = a.T
    b = np.cov(a)[-2]
    if i == 0:
        return_x = c
        coef_x = b
    else:
        return_x = np.vstack((return_x, c))
        coef_x = np.vstack((coef_x, b))

data6 = pd.DataFrame(data=data6)
data6['retx'] = pd.DataFrame(data=np.array([return_2]).T, columns=['a'])['a']
data6['codate'] = data2['codate']
date_x6 = list(set(data6['codate'].values))
date_x6.sort()
for i in range(len(date_x6)):
    a = data6[data6['codate'] == date_x6[i]]
    a.index = range(len(a['retx']))
    c = np.array([np.nanmean(a['retx'])])
    a = a.ix[:, :-1].values
    a = a.T
    b = np.cov(a)[-2]
    if i == 0:
        return_xy = c
        coef_xy = b
    else:
        return_xy = np.vstack((return_xy, c))
        coef_xy = np.vstack((coef_xy, b))

# 双选择Lasso
'''
alpha_ridge = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]
coeffs = {}
print(coef_x)
coef_x[np.isnan(coef_x)] = 0.0
for alpha in alpha_ridge:
    r = Lasso(alpha=alpha, normalize=True, max_iter=1000000)
    r = r.fit(coef_x, return_x)

grid_search = GridSearchCV(Lasso(alpha=1, normalize=True), scoring='neg_mean_squared_error',
                           param_grid={'alpha': alpha_ridge}, cv=10, n_jobs=-1)
grid_search.fit(coef_x, return_x)
alpha = alpha_ridge
rmse = list(np.sqrt(-grid_search.cv_results_['mean_test_score']))
print(rmse)
plt.figure(figsize=(6, 5))
lasso_cv = pd.Series(rmse, index=alpha)
lasso_cv.plot(title="Validation - LASSO", logx=True)
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()
'''
lasso = Lasso(alpha=1e-2, normalize=True, max_iter=1e6)
coef_x[np.isnan(coef_x)] = 0.0
return_x[np.isnan(return_x)] = 0.0
coef_xy[np.isnan(coef_xy)] = 0.0
return_xy[np.isnan(return_xy)] = 0.0

lasso = lasso.fit(coef_x, return_x)
x = sm.add_constant(coef_x)
y = return_x
coef = pd.Series(lasso.coef_)
coff_d = pd.DataFrame(data=coef, columns=['Lasso-coff'])

print('a', len(date_x))
data3 = coff_d
index_1 = list(np.where(np.abs(data3) != 0.0)[0])
print('index1', len(index_1))
data5[np.isnan(data5)] = 0.0
for j in range(np.shape(data5.values)[1] - 2):
    for i in range(len(date_x)):
        a = data5[data5['codate'] == date_x[i]]
        c = np.cov(a[[j, 'retx']].values.T)[0][1]
        a = a.ix[:, :-1].values.T
        b = np.cov(a)[-1]
        if i == 0:
            return_x = c
            coef_x = b
        else:
            return_x = np.vstack((return_x, c))
            coef_x = np.vstack((coef_x, b))

    lasso = Lasso(alpha=0.5, normalize=True, max_iter=1e6)
    lasso = lasso.fit(coef_x, return_x)

    coef = pd.Series(lasso.coef_)
    t1 = np.abs(np.array([coef.values]))
    if j == 0:
        lasso_s2r = t1
    else:
        lasso_s2r += t1
lasso_s2r = lasso_s2r[0][:-2]
# 因子筛选OLS

lasso_s2r[np.log(np.abs(lasso_s2r)) > 12] = 0
lasso_s2r[np.log(np.abs(lasso_s2r)) < 7] = 0

index_2 = list(np.where(lasso_s2r != 0.0)[0])
#index_3 = list(set(index_1) | set(index_2))

#for i in [68, 69, 71, 27, 29, 30, 35, 39, 48, 61, 63, 64, 65, 66, 72, 74]:
#    index_3.remove(i)

dataxx1 = data1.ix[:, list(set(index_1))]
dataxx2 = Sdata1.ix[:, list(set(index_2))]
dataxx=pd.concat([dataxx1,dataxx2],axis=1,join_axes=dataxx1.index)
dataxx.to_csv('watch.csv', index=False)

print('最后的因子为: ', len(index_3))
print(list(range(len(index_3))))
print(index_3)
data6 = dataxx.copy(deep=True)
data7 = dataxx.copy(deep=True)
x = sm.add_constant(data6.values)
y = return_1
x[np.isnan(x)] = 0.0
y[np.isnan(y)] = 0.0
regr = sm.OLS(y, x)
res = regr.fit()

# 预测与回测
r_x = np.dot(res.params[1:].T, data6.values.T)
r_x += res.params[0]
bt_1['retx_f'] = pd.DataFrame(data=np.array([r_x]).T, columns=['a'])['a']

date = list(set(bt_1['codate']))
date.sort()
for i in range(len(date)):
    temp = bt_1[bt_1['codate'] == date[i]]
    temp.sort_values("retx_f", inplace=True, ascending=False)
    temp.index = range(len(temp['retx_f']))
    temp = temp[0:10]

    a = np.mean(temp['retx_f'])
    b = np.mean(temp['retx'])
    temp_d = np.array([a, b])
    if i == 0:
        bt_data = temp_d
    else:
        bt_data = np.vstack((bt_data, temp_d))

x6 = coef_xy.T[index_3].T
# print(np.shape(x6))
x = coef_x.T[index_3].T


# print(np.shape(x))

def kmo(dataset_corr):
    corr_inv = np.linalg.inv(dataset_corr)
    nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
    A = np.ones((nrow_inv_corr, ncol_inv_corr))
    for i in range(0, nrow_inv_corr, 1):
        for j in range(i, ncol_inv_corr, 1):
            A[i, j] = -(corr_inv[i, j]) / (np.abs((corr_inv[i, i] * corr_inv[j, j])) ** 0.5)
            A[j, i] = A[i, j]
    dataset_corr = np.asarray(dataset_corr)
    kmo_num = (np.sum(np.square(dataset_corr)) - np.sum(np.square(np.diagonal(A)))) * 2
    kmo_denom = kmo_num + (np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A))))**0.8
    kmo_value = (kmo_num / kmo_denom) ** 0.125
    return kmo_value


x[np.isnan(x)] = 0.0
print(kmo(np.corrcoef(x)))

'''
pca = PCA(n_components=18)
pca.fit(x)
xa = pca.transform(x)
xa6 = pca.transform(x6).T
'''
xa = x
xa6 = x6.T

# xa = np.delete(xa, [4, 5, 8, 9, 14, 15, 16], axis=1)
# xa6 = np.delete(xa6, [4, 5, 8, 9, 14, 15, 16], axis=1)


x = xa[1:]
y = bt_data[:, 1].T[:-1]
print(np.shape(x))
print(np.shape(y))
pd.DataFrame(data=x).to_csv('999.csv', index=False)

regr = sm.OLS(y, sm.add_constant(x))
res = regr.fit()
print(res.summary())
result_t1 = res.pvalues
result_t2 = len(result_t1[result_t1 < 0.1]) / len(result_t1)
print(result_t2)

# SVR部分
clf = SVR()
print(np.shape(xa6))
# y=y.reshape(1,-1)
clf.fit(x, y)

r_x = np.dot(res.params[1:].T, xa6)
r_x += res.params[0]

# SVR部分
r_x = clf.predict(xa6.T)

bt_16['retx_f'] = pd.DataFrame(data=np.array([r_x]).T, columns=['a'])['a']

date = list(set(bt_16['codate']))
date.sort()

# for fdvvf in range(49):
for i in range(len(date)):
    temp = bt_16[bt_16['codate'] == date[i]]
    temp.sort_values("retx_f", inplace=True, ascending=False)
    temp.index = range(len(temp['retx_f']))
    temp = temp[0:12 + 1]
    # temp = temp[0:3]
    a = np.mean(temp['retx_f'])
    b = np.mean(temp['retx'])
    temp_d = np.array([a, b])
    if i == 0:
        bt_data = temp_d
    else:
        bt_data = np.vstack((bt_data, temp_d))

renturn_x = [1.0]
for i in range(len(date) - 1):
    if bt_data[i][0] >= 0:
        renturn_x.append(renturn_x[-1] * (1 + bt_data[i + 1][1]))
    else:
        renturn_x.append(renturn_x[-1])

print(renturn_x[-1])
print(renturn_x)

plt.plot(range(len(renturn_x)), renturn_x, label='Another line', linewidth=3, color='r', marker='o',
         markerfacecolor='blue', markersize=12)
plt.xlabel('period')
plt.ylabel('value')
plt.title('Lasso Backtest')
plt.legend()
plt.show()
