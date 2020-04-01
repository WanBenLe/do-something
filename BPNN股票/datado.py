# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 04:13:11 2017

@author: Administrator
"""
import pandas as pd
import numpy as np
STOCKP=['SHSE.600000', 'SHSE.600008', 'SHSE.600009', 'SHSE.600010', 'SHSE.600015', 'SHSE.600016', 'SHSE.600018', 'SHSE.600023', 'SHSE.600028', 'SHSE.600029', 'SHSE.600030', 'SHSE.600031', 'SHSE.600036', 'SHSE.600037', 'SHSE.600038', 'SHSE.600048', 'SHSE.600060', 'SHSE.600061', 'SHSE.600066', 'SHSE.600068', 'SHSE.600074', 'SHSE.600085', 'SHSE.600089', 'SHSE.600104', 'SHSE.600109', 'SHSE.600111', 'SHSE.600115', 'SHSE.600118', 'SHSE.600150', 'SHSE.600157', 'SHSE.600170', 'SHSE.600177', 'SHSE.600188', 'SHSE.600196', 'SHSE.600208', 'SHSE.600221', 'SHSE.600256', 'SHSE.600276', 'SHSE.600297', 'SHSE.600309', 'SHSE.600332', 'SHSE.600340', 'SHSE.600352', 'SHSE.600362', 'SHSE.600369', 'SHSE.600372', 'SHSE.600373', 'SHSE.600376', 'SHSE.600383', 'SHSE.600415', 'SHSE.600436', 'SHSE.600446', 'SHSE.600482', 'SHSE.600489', 'SHSE.600498', 'SHSE.600518', 'SHSE.600519', 'SHSE.600522', 'SHSE.600535', 'SHSE.600547', 'SHSE.600549', 'SHSE.600570', 'SHSE.600583', 'SHSE.600585', 'SHSE.600588', 'SHSE.600606', 'SHSE.600637', 'SHSE.600660', 'SHSE.600674', 'SHSE.600682', 'SHSE.600685', 'SHSE.600688', 'SHSE.600690', 'SHSE.600703', 'SHSE.600704', 'SHSE.600705', 'SHSE.600718', 'SHSE.600737', 'SHSE.600739', 'SHSE.600741', 'SHSE.600795', 'SHSE.600804', 'SHSE.600816', 'SHSE.600820', 'SHSE.600827', 'SHSE.600837', 'SHSE.600871', 'SHSE.600886', 'SHSE.600887', 'SHSE.600893', 'SHSE.600895', 'SHSE.600900', 'SHSE.600958', 'SHSE.600959', 'SHSE.600999', 'SHSE.601006', 'SHSE.601009', 'SHSE.601018', 'SHSE.601021', 'SHSE.601088', 'SHSE.601099', 'SHSE.601111', 'SHSE.601117', 'SHSE.601118', 'SHSE.601155', 'SHSE.601166', 'SHSE.601169', 'SHSE.601186', 'SHSE.601198', 'SHSE.601211', 'SHSE.601216', 'SHSE.601225', 'SHSE.601288', 'SHSE.601318', 'SHSE.601328', 'SHSE.601333', 'SHSE.601336', 'SHSE.601377', 'SHSE.601390', 'SHSE.601398', 'SHSE.601555', 'SHSE.601600', 'SHSE.601601', 'SHSE.601607', 'SHSE.601618', 'SHSE.601628', 'SHSE.601633', 'SHSE.601668', 'SHSE.601669', 'SHSE.601688', 'SHSE.601718', 'SHSE.601766', 'SHSE.601788', 'SHSE.601800', 'SHSE.601818', 'SHSE.601857', 'SHSE.601866', 'SHSE.601877', 'SHSE.601888', 'SHSE.601899', 'SHSE.601901', 'SHSE.601919', 'SHSE.601933', 'SHSE.601939', 'SHSE.601958', 'SHSE.601985', 'SHSE.601988', 'SHSE.601989', 'SHSE.601992', 'SHSE.601998', 'SHSE.603993', 'SZSE.000001', 'SZSE.000008', 'SZSE.000009', 'SZSE.000060', 'SZSE.000063', 'SZSE.000069', 'SZSE.000156', 'SZSE.000157', 'SZSE.000166', 'SZSE.000333', 'SZSE.000338', 'SZSE.000402', 'SZSE.000423', 'SZSE.000425', 'SZSE.000540', 'SZSE.000559', 'SZSE.000568', 'SZSE.000623', 'SZSE.000625', 'SZSE.000627', 'SZSE.000630', 'SZSE.000671', 'SZSE.000686', 'SZSE.000709', 'SZSE.000725', 'SZSE.000728', 'SZSE.000738', 'SZSE.000750', 'SZSE.000768', 'SZSE.000776', 'SZSE.000783', 'SZSE.000792', 'SZSE.000793', 'SZSE.000826', 'SZSE.000839', 'SZSE.000858', 'SZSE.000876', 'SZSE.000895', 'SZSE.000938', 'SZSE.000959', 'SZSE.000961', 'SZSE.000963', 'SZSE.000977', 'SZSE.000983', 'SZSE.001979', 'SZSE.002007', 'SZSE.002008', 'SZSE.002024', 'SZSE.002027', 'SZSE.002074', 'SZSE.002081', 'SZSE.002131', 'SZSE.002142', 'SZSE.002146', 'SZSE.002152', 'SZSE.002153', 'SZSE.002174', 'SZSE.002183', 'SZSE.002195', 'SZSE.002202', 'SZSE.002236', 'SZSE.002241', 'SZSE.002292', 'SZSE.002304', 'SZSE.002310', 'SZSE.002385', 'SZSE.002411', 'SZSE.002415', 'SZSE.002424', 'SZSE.002450', 'SZSE.002456', 'SZSE.002466', 'SZSE.002470', 'SZSE.002475', 'SZSE.002500', 'SZSE.002508', 'SZSE.002594', 'SZSE.002673', 'SZSE.002714', 'SZSE.002736', 'SZSE.300017', 'SZSE.300024', 'SZSE.300033', 'SZSE.300059', 'SZSE.300070', 'SZSE.300072', 'SZSE.300124', 'SZSE.300133', 'SZSE.300144']
print('成份股数目:'+str(len(STOCKP)))
dfos='E:\\0AAA\\'+STOCKP[0]+'.csv'
df01=pd.read_csv(open(dfos))
for a in range(len(STOCKP)-1):
    dfos='E:\\0AAA\\'+STOCKP[a+1]+'.csv'
    df02=pd.read_csv(open(dfos))
    df01=pd.concat([df01,df02]) 
del df01['Unnamed: 0']
df01.to_csv('E:\\0AAA\\df01.csv')
dfos='E:\\0AAA\\df01.csv'
df01=pd.read_csv(open(dfos))
del df01['Unnamed: 0']
df01=df01.sort_values(by='cdate')
df01.to_csv('E:\\0AAA\\df02.csv')
dfos='E:\\0AAA\\df02.csv'
df01=pd.read_csv(open(dfos))
del df01['Unnamed: 0']
outputfile = 'output.xls' 
modelfile = 'modelweight.model' 
data =df01
print(len(data['ma3']))   
feature = ['ma3','PER','PBR','PSR','EPS','bvps','cfps','afps'] 
#因子
#训练集
stockdf=pd.DataFrame(columns=['ldate','return','stock','ma3','PER','PBR','PSR','EPS','bvps','cfps','afps'],index=np.arange(len(data['ma3'])))
for b in feature:
    dmean=np.mean(data[b])
    dstd=np.std(data[b])
    for c in range(len(data['ma3'])):
        if data[b][c]>dmean+3*dstd:
            stockdf[b][c]=dmean+3*dstd
        elif data[b][c]<dmean-3*dstd:
            stockdf[b][c]=dmean-3*dstd
        else:
            stockdf[b][c]=data[b][c]
for d in range(len(data['ma3'])):
    stockdf['ldate'][d]=data['ldate'][d] 
    stockdf['stock'][d]=data['stock'][d] 
    stockdf['return'][d]=data['return'][d] 
stockdf.to_csv('E:\\0AAA\\pretrain.csv')
feature = ['ma3','PER','PBR','PSR','EPS','bvps','cfps','afps'] 
train=pd.DataFrame(columns=['ldate','rank','ma3','PER','PBR','PSR','EPS','bvps','cfps','afps'],index=np.arange(len(data['ma3'])))
for b in feature:
    dmean=np.mean(data[b])
    dstd=np.std(data[b])
    for c in range(len(data['ma3'])):
        train[b][c]=(stockdf[b][c]-dmean)/(dstd)
for d in range(len(data['ma3'])):
    train['ldate'][d]=stockdf['ldate'][d] 
    if (stockdf['return'][d]>(1/15)):
        train['rank'][d]=4
    elif (stockdf['return'][d]>(1/30)):
        train['rank'][d]=3
    elif (stockdf['return'][d]>(-1/30)):
        train['rank'][d]=2
    elif (stockdf['return'][d]>(-1/15)):
        train['rank'][d]=1
    else:
        train['rank'][d]=0
train.to_csv('E:\\0AAA\\0train.csv')
print(train)