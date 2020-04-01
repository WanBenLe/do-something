import pandas as pd
import numpy as np
import statsmodels.api as sm
from numba import jit

data3 = pd.read_csv('data2.csv')
data3_1 = data3[data3['rank'] == 1][['aver', 'aveBP']]
x = sm.add_constant(data3_1[['aveBP']])
y = data3_1['aver']
regr = sm.OLS(y, x)
res = regr.fit()
print(res.summary())

data3_2 = data3[data3['rank'] == 10][['aver', 'aveBP']]
x = sm.add_constant(data3_2[['aveBP']])
y = data3_2['aver']
regr = sm.OLS(y, x)
res = regr.fit()
print(res.summary())


# read data
data0 = pd.read_csv('data.csv')
# get stockid and date and sort
stockid = list(set(data0['PERMNO'].values))
stockid.sort()

date = list(set(data0['date'].values))
date.sort()
# cal ln(PB^-1)
data0['lnBP'] = np.log(data0['PRC'] / data0['SHROUT'] / data0['ASK'] * 10000)
# del column PRC and SHROUT
data0 = data0.drop(columns=['PRC', 'SHROUT'])


@jit
# data cleaning
def nand(ser):
    for i in range(len(ser)):
        if np.isnan(ser[i]):
            ser[i] = 0.0
    return ser


# cal ave return
def ave_r(g1, d1, d2):
    r0 = 0
    k0 = 0
    for i in g1:
        p1 = data0[(data0['PERMNO'] == i) & (data0['date'] == d2)]['ASK'].values
        p2 = data0[(data0['PERMNO'] == i) & (data0['date'] == d1)]['ASK'].values
        r1 = (p2 - p1) / p1
        if (not np.isnan(r1)) and r1.size > 0:
            try:
                r0 += r1
                k0 += 1
            except:
                'data loss'
    r0 /= k0
    return r0


data0['lnBP'] = nand(data0['lnBP'].values)
hula = len(date) * 10
# by date
for i in range((len(date) - 1)):
    d_temp = date[i]
    data1 = data0[data0['date'] == d_temp]
    data1 = data1.sort_values(by=["lnBP"], ascending=False)
    numx = np.floor(len(data1) / 10)
    # by rank cal ave return and PB^-1
    for j in range(10):
        print(((i + 1) * 10 + j + 1) / hula * 100)
        g1 = data1[int(j * numx):int((j + 1) * numx)]['PERMNO'].values
        meanreturn = ave_r(g1, date[i], date[i + 1])
        meanBP = np.mean(data1[int(j * numx):int((j + 1) * numx)]['lnBP'])
        temp_r = np.array([d_temp, j + 1, meanreturn, meanBP])
        if i == 0 and j == 0:
            rexxx = temp_r
        else:
            rexxx = np.vstack((rexxx, temp_r))
data2 = pd.DataFrame(data=rexxx, columns=['date', 'rank', 'aver', 'aveBP'])
data2.to_csv('data2.csv', index=False)
print(data2)
