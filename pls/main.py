# coding utf-8
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cross_decomposition import PLSRegression
import os
from math import factorial


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except:
        print(ValueError("阶数求导窗口要整数"))
        os._exit(-1)
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("窗口需要为正奇整数")
    if window_size < order + 2:
        raise TypeError("窗口数太小")
    order_range = range(order)
    half_window = (window_size - 1) // 2
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


f_para = open("para.txt", "r")
line_para = f_para.readlines()
# 读取参数
par_1 = int(line_para[0])
par_2 = int(line_para[1])
par_3 = int(line_para[2])
par_4 = int(line_para[3])
par_5 = int(line_para[4])
par_6 = int(line_para[5])
f_para.close()

data0 = pd.read_csv('data.csv').values
Y = data0[0].reshape(-1, 1)
data0 = data0[1:].T
X = pd.read_csv('X.csv').values.T
s1 = np.shape(data0)[1]
s2 = np.shape(X)[1]

if par_6 > par_3 and par_3 > 0:
    print('PLS维数不能高于PCA的维数!')
    os._exit(-1)
elif s1 != s2:
    print('训练和预测的X维数必须相同!')
    os._exit(-1)
elif par_3 > s1:
    print('PCA.PLS的维数不能大于训练和预测X的维数')
    os._exit(-1)

if par_1 != 0:
    data1 = data0.T
    for i in range(data1.shape[0]):
        data1[i] = (data1[i] - np.min(data1[i])) / (np.max(data1[i]) - np.min(data1[i]))
    data1 = data1.T
else:
    data1 = data0

# Z标准化
if par_2 != 0:
    data2 = data1.T
    for i in range(data2.shape[0]):
        temp = (data2[i].reshape(1, -1) - np.mean(data2[i].reshape(1, -1))) / np.std(data2[i].reshape(1, -1))
        data2[i] = temp
    data2 = data2.T
else:
    data2 = data1

# PCA
if par_3 != 0:
    data3 = data1
    pca = PCA(n_components=par_3)
    data3 = pca.fit(data3).transform(data3)
else:
    data3 = data2

if par_4 != 0 and par_5 != 0:
    if par_4 == 1:
        for i in range(np.shape(data3)[0]):
            temp_1 = savitzky_golay(data3[i], window_size=par_5, order=1)
            if i == 0:
                temp_2 = temp_1
            else:
                temp_2 = np.vstack((temp_2, temp_1))
        data4 = temp_2
    elif par_4 == 2:
        for i in range(np.shape(data3)[0]):
            temp_1 = savitzky_golay(data3[i], window_size=par_5, order=2)
            if i == 0:
                temp_2 = temp_1
            else:
                temp_2 = np.vstack((temp_2, temp_1))
        data4 = temp_2
    else:
        data4 = data3
else:
    data4 = data3
pls1 = PLSRegression(n_components=par_6)
pls1.fit(data4, Y)
print('a')
print(np.shape(data4))
print(np.shape(Y))
Y_pred = pls1.predict(data4)
print(np.shape(Y_pred))
print('R^2', pls1.score(data4, Y, sample_weight=None))

# 预测

if par_1 != 0:
    data1 = X.T
    for i in range(data1.shape[0]):
        data1[i] = (data1[i] - np.min(data1[i])) / (np.max(data1[i]) - np.min(data1[i]))
    data1 = data1.T
else:
    data1 = X

# Z标准化
if par_2 != 0:
    data2 = data1.T
    for i in range(data2.shape[0]):
        temp = (data2[i].reshape(1, -1) - np.mean(data2[i].reshape(1, -1))) / np.std(data2[i].reshape(1, -1))
        data2[i] = temp
    data2 = data2.T
else:
    data2 = data1

# PCA
if par_3 != 0:
    data3 = data1
    pca = PCA(n_components=par_3)
    data3 = pca.fit(data3).transform(data3)
else:
    data3 = data2
print(np.shape(data3))
print(data3[0])
if par_4 != 0 and par_5 != 0:
    if par_4 == 1:
        for i in range(np.shape(data3)[0]):
            temp_1 = savitzky_golay(data3[i], window_size=par_5, order=1)
            if i == 0:
                temp_2 = temp_1
            else:
                temp_2 = np.vstack((temp_2, temp_1))
        data4 = temp_2
    elif par_4 == 2:
        for i in range(np.shape(data3)[0]):
            temp_1 = savitzky_golay(data3[i], window_size=par_5, order=2)
            if i == 0:
                temp_2 = temp_1
            else:
                temp_2 = np.vstack((temp_2, temp_1))
        data4 = temp_2
else:
    data4 = data3

Y_pred = pls1.predict(data4.reshape(1,-1))
print(np.shape(Y_pred))
pd.DataFrame(data4).to_csv('result.csv', index=False)
pd.DataFrame(Y_pred).to_csv('forcast.csv', index=False)
print('finished!')
