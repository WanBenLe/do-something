import numpy as np
import pandas as pd
from numba import jit
from sklearn.preprocessing import scale


def TrSAXCreat(tseries):
    omega = 300
    jie_x = np.zeros((len(tseries) - omega, omega))
    duanshu = 100
    duan = int(omega / duanshu)
    duan_m = np.zeros((len(tseries) - omega, duan))
    duan_k = np.zeros((len(tseries) - omega, duan))
    for i in range(len(tseries) - omega):
        jie_x[i] = tseries[i:i + omega]
        for j in range(int(duan)):
            duan_m[i, j] = np.mean(jie_x[i, j:j + 1])
            y1 = jie_x[i, j:j + 1]
            x1 = np.arange(0, duanshu) + duanshu * i
            duan_k[i, j] = (np.mean(x1 * y1) - np.mean(y1) * np.mean(x1)) / (np.mean(x1 ** 2) - np.mean(x1) ** 2)

    zimu1 = np.zeros_like(duan_m, dtype=object)
    zimu2 = np.zeros_like(duan_m, dtype=object)
    duan_m = scale(duan_m)
    '''
    gridx1 = [-np.inf, np.percentile(duan_m.reshape(-1), 25), np.percentile(duan_m.reshape(-1), 50),
              np.percentile(duan_m.reshape(-1), 75), np.inf]
    '''
    gridx1 = [-np.inf, -0.43, 0.43, np.inf]
    gridx2 = ['a', 'b', 'c']
    for i in range(len(gridx2)):
        zimu1[(duan_m > gridx1[i]) & (duan_m <= gridx1[i + 1])] = gridx2[i]
    zimu2[duan_k > 1] = 'E'
    zimu2[(duan_k <= 1) & (duan_k > 0)] = 'D'
    zimu2[duan_k == 0] = 'C'
    zimu2[(duan_k >= -1) & (duan_k < 0)] = 'B'
    zimu2[duan_k < -1] = 'A'
    TrSAX = zimu2[:] + zimu1[:]
    TrSAX = TrSAX[:, 0] + TrSAX[:, 1] + TrSAX[:, 2]
    indx = 1
    while len(TrSAX) > indx:
        if TrSAX[indx] == TrSAX[indx - 1]:
            TrSAX = np.delete(TrSAX, indx, axis=0)
        else:
            indx += 1

    return TrSAX


tseries1 = np.random.rand(1000)
TrSAX1 = TrSAXCreat(tseries1)
tseries2 = np.random.rand(1000)
TrSAX2 = TrSAXCreat(tseries2)

df1 = pd.DataFrame(TrSAX1.reshape(-1, 1), columns=['strx'])
df1['a'] = TrSAX1.reshape(-1, 1)
df1 = pd.pivot_table(df1, index=['strx'], values=['a'], aggfunc='count', margins=True).iloc[:-1]
df1['cols'] = list(df1.index)

df2 = pd.DataFrame(TrSAX2.reshape(-1, 1), columns=['strx'])
df2['a'] = TrSAX2.reshape(-1, 1)
df2 = pd.pivot_table(df2, index=['strx'], values=['a'], aggfunc='count', margins=True).iloc[:-1]
df2['cols'] = list(df2.index)

df3 = pd.merge(df1, df2, how='outer', left_on='cols', right_on='cols').fillna(0)[['cols', 'a_x', 'a_y']]

Dis = np.sum((df3['a_x'].values - df3['a_y'].values) ** 2) ** 0.5
print(Dis)
print(1)
