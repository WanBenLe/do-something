import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings

'''
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)
'''
warnings.filterwarnings("ignore")


# 过去12个月的数据data1估计beta放在data0
def beta_cal(data0, data1):
    data0['exr'] = data0['rx'] - data0['rf']
    data1['exr'] = data1['rx'] - data1['rf']
    data0['rmx'] = data0['rm'] - data0['rf']
    id_list = list(set(data0['id']))
    id_list.sort()
    date_list = list(set(data0['date']))
    date_list.sort()

    data0['beta'] = 0.0

    for i in id_list:
        df_temp = data1[data1['id'] == i]
        x = sm.add_constant(df_temp['exr'].values)
        y = df_temp['rm'].values
        y[np.isnan(y)] = np.nanmean(y)
        x[np.isnan(x)] = np.nanmean(x)
        regr = sm.OLS(y, x)
        res = regr.fit()
        data0[data0['id'] == i]['beta'] = res.params[1]
    return data0, date_list


# 构建回归用的因子,若show_re=1,估计model
def Buff(data0, date_list, show_re):
    for i in date_list:
        df_temp = data0[data0['date'] == i]
        df_temp.index = list(range(len(df_temp['X3'])))
        num_temp = int(len(df_temp['X3']) / 10)
        df_temp.sort_values(by='X3', ascending=True)
        SMB = np.nanmean(df_temp[:num_temp]['rx']) - np.nanmean(df_temp[-num_temp:]['rx'])
        df_temp.sort_values(by='X2', ascending=True)
        HML = np.nanmean(df_temp[:num_temp]['rx']) - np.nanmean(df_temp[-num_temp:]['rx'])
        MKT = np.nanmean(df_temp['rmx'])
        df_temp.sort_values(by='X4', ascending=False)
        UMD = np.nanmean(df_temp[:num_temp]['rx']) - np.nanmean(df_temp[-num_temp:]['rx'])
        df_temp.sort_values(by='beta', ascending=True)
        BAB = np.nanmean(df_temp[:num_temp]['rx']) - np.nanmean(df_temp[-num_temp:]['rx'])
        df_temp.sort_values(by='X1', ascending=False)
        QMJ = np.nanmean(df_temp[:num_temp]['rx']) - np.nanmean(df_temp[-num_temp:]['rx'])
        Y_temp = np.nanmean(df_temp['exr'])
        X_temp = np.array([MKT, SMB, HML, UMD, BAB, QMJ])
        if i == date_list[0]:
            Y = Y_temp
            X = X_temp
        else:
            Y = np.vstack((Y, Y_temp))
            X = np.vstack((X, X_temp))
    if show_re == 1:
        x = sm.add_constant(X[:-1])
        y = Y[1:]
        regr = sm.OLS(y, x)
        res = regr.fit()
        print(res.summary())
        return X, res.params
    else:
        return X


def backtest(r_back, r_for, plt_show):
    if len(r_back) != len(r_for):
        print('Shape Error!')
        print(np.shape(r_back))
        print(np.shape(r_for))
        return -1
    else:
        rxx = [1]
        for i in range(len(r_back)):
            if r_for[i] > 0:
                rxx.append(rxx[-1] * (r_back[i]+1))
            else:
                rxx.append(rxx[-1])
        if plt_show == 1:
            plt.plot(range(len(rxx)), rxx, label='Return', linewidth=3, color='r', marker='o',
                     markerfacecolor='blue', markersize=12)
            plt.xlabel('period')
            plt.ylabel('value')
            plt.title('Backtest')
            plt.legend()
            plt.show()
        return rxx


data0 = pd.read_csv('cndata.csv')
data1 = pd.read_csv('cndata.csv')
data2 = pd.read_csv('cndata.csv')
# 此处是个小例子,用data1估计beta,data0估计model
data0, date_list = beta_cal(data0, data1)
factors, para = Buff(data0, date_list, show_re=1)

# 然后data0估计beta,在data2上构建因子,并回测
data2, date_list = beta_cal(data2, data0)
factors = Buff(data2, date_list, show_re=0)
r_for = np.dot(para[1:].reshape(1, -1), factors.T) + para[0]
r_for=r_for.T
# 构建回测序列,用的是模型等同的超额收益率
for i in date_list:
    df_temp = data2[data2['date'] == i]
    if i == date_list[0]:
        r_back = np.nanmean(df_temp['exr'])
    else:
        r_back = np.vstack((r_back, np.nanmean(df_temp['exr'])))

r_back = r_back
rxx = backtest(r_back, r_for, plt_show=1)
print(rxx)
