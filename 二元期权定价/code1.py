import pandas as pd
import numpy as np
from numba import jit, int32
from matplotlib import pyplot as plt
import time

data0 = pd.read_csv('volume.csv')
print(data0)
data1 = data0['time'].values
data2 = data0[['volume', 'day']].values


@jit
def vol_x(data1, data2):
    volume_x = np.array([])
    for i in range(len(data1)):
        if i == 0:
            volume_x = np.array([data2[i][0]])
        elif data1[i] == data1[i - 1] and data2[i][1] == data2[i - 1][1]:
            volume_x[-1] += data2[i][0]
        else:
            volume_x = np.vstack((volume_x, np.array(data2[i][0])))
    return volume_x


vol_0 = vol_x(data1, data2)
print(vol_0)


@jit
def vol_y(vol_0):
    num_x = 15
    volume_y = np.array([])
    k = int(len(vol_0) / num_x)
    for i in range(k):
        if i == 0:
            volume_y = np.array([np.sum(vol_0[num_x * i:num_x * (i + 1)])])
        else:
            volume_y = np.vstack((volume_y, np.array(np.sum(vol_0[num_x * i:num_x * (i + 1)]))))
    return volume_y


vol_1 = vol_y(vol_0)
print(vol_1)


@jit
def vol_z(vol_1):
    all_vol = np.sum(vol_1)
    for i in range(len(vol_1)):
        vol_1[i] = int(vol_1[i] / all_vol * 100000)
    return vol_1


vol_2 = vol_z(vol_1)

plt.bar(range(len(vol_2)), vol_2.T[0].tolist())
plt.show()
