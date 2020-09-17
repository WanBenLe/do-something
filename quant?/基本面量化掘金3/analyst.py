# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import copy
import numpy as np
from numba import jit
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from statsmodels.api import OLS, add_constant
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import gc

# 设置token
set_token('b81189eb81a34b3fe92e3ee339430f9fafb87d68')


class runpostlasso():
    def delete_rows(dfData):
        # dfData:预处理数据，df类型
        newData = dfData.dropna(axis=0, how='any').reset_index(drop=True)  # 清洗数据
        return newData

    def __init__(self, contextnow,trade_count):
        self.error_code = 0
        self.para_name = ''
        self.params = np.array([0.0])
        self.name_list = []
        f_data = u"DY,EV,EVEBITDA,EVPS,LYDY,NEGOTIABLEMV,PB,PCLFY,PCTTM,PELFY,PELFYNPAAEI,PEMRQ,PEMRQNPAAEI,PETTM,PETTMNPAAEI,PSLFY,PSMRQ,PSTTM,TCLOSE,TOTMKTCAP,TURNRATE,TOTAL_SHARE,FLOW_SHARE"
        f_dataname = ['DY', 'EV', 'EVEBITDA', 'EVPS', 'LYDY', 'NEGOTIABLEMV', 'PB', 'PCLFY', 'PCTTM', 'PELFY',
                      'PELFYNPAAEI', 'PEMRQ', 'PEMRQNPAAEI', 'PETTM', 'PETTMNPAAEI', 'PSLFY', 'PSMRQ', 'PSTTM',
                      'TCLOSE',
                      'TOTMKTCAP', 'TURNRATE', 'TOTAL_SHARE', 'FLOW_SHARE']

        stock300 = list(get_history_constituents(index='SHSE.000016', start_date=contextnow,
                                                 end_date=contextnow)[0]['constituents'].keys())
        print('成份股:', stock300)
        conutx = trade_count*3
        for stock in stock300:
            fin = get_fundamentals_n(table='trading_derivative_indicator', symbols=stock,
                                     end_date=contextnow, fields=f_data, df=True, count=conutx)
            '''
            # 加上平方项做非线性调整
            for i in range(len(f_dataname)):
                fin[f_dataname[i] + 'squre'] = fin[f_dataname[i]].values ** 2
            '''
            # 查询历史行情
            close = \
                history_n(symbol=stock, frequency='1d', end_time=contextnow, fields='close', count=conutx + 1, adjust=1,
                          df=True)[
                    'close'].values
            return_day = (np.diff(close) / close[0:-1])[1:]
            fin = fin.iloc[0:-1]
            # 如果财务比停牌的多,那就用最新的财务去截断
            if len(fin) > return_day.shape[0]:
                fin = fin.iloc[-return_day.shape[0]:]
            fin['return'] = return_day
            if stock == stock300[0]:
                all_data = fin
            else:
                all_data = pd.concat((all_data, fin))
        # all_data.to_csv('data.csv', index=False)
        # 删除缺失行
        all_data = runpostlasso.delete_rows(all_data)
        # all_data.to_csv('data_run.csv', index=False)

        # all_data=pd.read_csv('data_run.csv').values[:,3:]
        del all_data['symbol'], all_data['pub_date'], all_data['end_date']
        X_test = all_data['return'].values
        del all_data['return']
        all_data = all_data.values
        # all_data = all_data.values[:, 3:]
        # 最大最小缩放
        try:
            all_data = MinMaxScaler().fit_transform(all_data)
        except:
            print(1)
        # pd.DataFrame(all_data).to_csv('data_run1.csv', index=False)
        gc.collect()

        # all_data = pd.read_csv('data_run1.csv')

        # all_data = all_data.values
        X_train = all_data

        # print(X_train.shape)
        # print('Lasso')
        coefs = LassoCV(max_iter=2000).fit(X_train, X_test).coef_
        # 获取Lasso不为0的列
        X_train = X_train[:, coefs != 0]
        if X_train.shape[1] == 0:
            self.error_code = -1
        else:
            F = mutual_info_regression(X_train, X_test)
            # print('回归信息熵的F值', F)
            # 取F<0.1的数据
            X_train = X_train[:, F < 0.1]
            # print(X_train.shape)
            if X_train.shape[1] == 0:
                self.error_code = -1
            else:
                self.name_list = np.array(f_dataname)[coefs != 0][F < 0.1]
                self.para_name = ",".join(self.name_list)
                X_train = add_constant(X_train).astype(float)
                X_test = X_test.astype(float)
                # OLS的得到t,用newy-west调整自相关与异方差(运算量问题~)
                model = OLS(X_train, X_test).fit()
                self.params = model.params.reshape(-1)
