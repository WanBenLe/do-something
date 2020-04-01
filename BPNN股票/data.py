# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:44:53 2017

@author: Administrator
"""
import time
#做日期-数字的变换
from predictive_imputer import predictive_imputer
#插补的库
#imputer = predictive_imputer.PredictiveImputer(f_model="RandomForest","KNN","PCA")
#X_trans = imputer.fit(X).transform(X.copy())

from gmsdk import md
import pandas as pd
import numpy as np
from collections import Counter 
#计算缺失值
global d
md.init('name', 'pwd')
k=md.get_constituents('SHSE.000300')
#上证50
K=[Constituent.symbol for Constituent in k]
STOCKP=[Constituent.symbol for Constituent in k]
print(len(K))
#获取成份股
lenindex=len(K)
#成份股数目
calendar=md.get_calendar('SHSE','2016-02-01','2017-06-21')
Calendar=[TradeDate.strtime for TradeDate in calendar]
ntradedays=len(Calendar)
print('tradedays:'+str(ntradedays))
#交易日天数
#,'total_asset','current_asset','fixed_asset','liability','current_liability','longterm_liability','equity','income','operating_profit','net_profit'
for a in range(lenindex):
    daily_bars= md.get_dailybars(K[a],'2016-02-01','2017-06-21')
    #daily_bars.reverse()
    #不需要
    market_index=md.get_market_index(K[a],'2016-02-01','2017-06-21')
    #market_index.reverse()
    #不需要
    financial_index=md.get_financial_index(K[a],'2015-12-01','2017-06-21')
    #开始日期选上一个有报告的时间
    #financial_index.reverse()
    #不需要
    date=[bar.strtime for bar in daily_bars]
    close=[bar.close for bar in daily_bars]
    adj_factor=[bar.adj_factor for bar in daily_bars]
    lenadj_factor=len(adj_factor)
    #获取复权因子数量
    PER=[MarketIndex.pe_ratio for MarketIndex in market_index]
    PBR=[MarketIndex.pb_ratio for MarketIndex in market_index]  
    PSR=[MarketIndex.ps_ratio for MarketIndex in market_index]
    EPS=[FinancialIndex.eps for FinancialIndex in financial_index]
    pub_date =[FinancialIndex.pub_date  for FinancialIndex in financial_index]
    pub_datedays=len(pub_date)
    pub_dateb=range(pub_datedays)
    for b in range(pub_datedays):
        pub_datea=time.strptime(pub_date[b], "%Y-%m-%d")
        pub_dateb[b]= int(time.mktime(pub_datea))  
    #把获取的财务报告日期化为数字
    bvps=[FinancialIndex.bvps for FinancialIndex in financial_index]                         #每股净资产  
    cfps=[FinancialIndex.cfps for FinancialIndex in financial_index]                        #每股现金流  
    afps=[FinancialIndex.afps for FinancialIndex in financial_index]                        #每股公积金  
    lenstock=len(close)
    #获取该股票有的数据条数
    stockdf=pd.DataFrame(columns=['ldate','cdate','date','stock','close','return','ma3','PER','PBR','PSR','EPS','bvps','cfps','afps'],index=np.arange(ntradedays))
    #pandas的数据框
    for e in range(ntradedays):
        stockdf['ldate'][e]=Calendar[e][0:10]
        datea=time.strptime(stockdf['ldate'][e], "%Y-%m-%d")
        dateb= int(time.mktime(datea))
        stockdf['cdate'][e]=dateb
        for t in range(lenstock):
            if stockdf['ldate'][e]==date[t][0:10]:
                stockdf['date'][e]=date[t][0:10]
                datea=time.strptime(stockdf['date'][e], "%Y-%m-%d")
                dateb= int(time.mktime(datea))        
                #把当天时间化为数字
                stockdf['close'][e]=close[t]*adj_factor[t]/adj_factor[lenadj_factor-1]
                #写入前复权价格
                if e>0:
                    if type(stockdf['close'][e-1]) == float and type(stockdf['close'][e-1]) !=0 :
                        stockdf['return'][e]=(stockdf['close'][e]-stockdf['close'][e-1])/stockdf['close'][e-1]
                    else:
                        stockdf['return'][e]='' 
                if e>2:
                    stockdf['ma3'][e]=(stockdf['return'][e]+stockdf['return'][e-1]+stockdf['return'][e-2])/3
              
                #计算收益率,第一个为0
                stockdf['PER'][e]=PER[t]
                stockdf['PBR'][e]=PBR[t]
                stockdf['PSR'][e]=PSR[t]
        d=-1
        for c in range(pub_datedays):
            if dateb>pub_dateb[c-1]:
                d=d+1
                #确定当前财务数据用哪个报告期,以最靠近当前时间但在前面的报告期为准
        stockdf['stock'][e]=K[a]
        stockdf['EPS'][e]=EPS[d]
        stockdf['bvps'][e]=bvps[d]
        stockdf['cfps'][e]=cfps[d]
        stockdf['afps'][e]=afps[d]
    f=list(stockdf['close'])
    g=np.array([list(stockdf['return']),list(stockdf['ma3']),list(stockdf['PER']),list(stockdf['PBR']),list(stockdf['PSR'])]).transpose()
    #需要插补的数据
    h=['return','ma3','PER','PBR','PSR']
    imputer = predictive_imputer.PredictiveImputer(f_model="RandomForest")
    X_trans = (imputer.fit(g).transform(g.copy())).transpose()
    #插补的方法,暂时是随机森林
    for l in range(len(h)):
        X_list= X_trans[l].tolist()
        stockdf[h[l]]=X_list
    del stockdf['date']
    del stockdf['close']
    if (Counter(f)[np.nan])<(ntradedays*0.1):
    #缺失率大于10%的就不要了    
        stockdf.to_csv(K[a]+'.csv')
        print('ok:'+K[a])
    else:
        print('out:'+K[a])
        STOCKP.remove(K[a])
        #写出CSV
    
print("finish")
print(STOCKP)