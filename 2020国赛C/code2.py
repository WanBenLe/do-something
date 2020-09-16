import pandas as pd
import numpy as np
from scipy import stats

df1_1 = pd.read_csv('企业信息.csv', encoding='ansi')[['企业代号', '信誉评级', '是否违约']].values
df1_2 = pd.read_csv('有进项.csv', encoding='ansi')[['企业代号', '金额', '税额', '价税合计', '发票状态']].values
df1_2 = df1_2[(df1_2[:, -1] == '有效发票'), :]
df1_3 = pd.read_csv('有销项.csv', encoding='ansi')[['企业代号', '金额', '税额', '价税合计', '发票状态']].values
df1_3 = df1_3[(df1_3[:, -1] == '有效发票'), :]

df2_1 = pd.read_csv('无企业信息.csv', encoding='ansi')[['企业代号']].values
df2_2 = pd.read_csv('无进项.csv', encoding='ansi')[['企业代号', '金额', '税额', '价税合计', '发票状态']].values
df2_2 = df2_2[(df2_2[:, -1] == '有效发票'), :]
df2_3 = pd.read_csv('无销项.csv', encoding='ansi')[['企业代号', '金额', '税额', '价税合计', '发票状态']].values
df2_3 = df2_3[(df2_3[:, -1] == '有效发票'), :]


# 特征构造函数
def tezheng(data1, data2, data3):
    qiye = np.unique(data1[:, 0])
    for dange in qiye:
        tdata_2 = data2[(data2[:, 0] == dange)]
        tdata_3 = data3[(data3[:, 0] == dange)]
        tdata_3[:, 3] = tdata_3[:, 3].astype(float)
        # 企业,
        try:
            tempx = np.array([dange, np.nansum(tdata_2[:, 1]), np.nansum(tdata_2[:, 2]), np.nansum(tdata_2[:, 3]),
                              np.nansum(tdata_3[:, 1]), np.nansum(tdata_3[:, 2]), np.nansum(tdata_3[:, 3]),
                              np.mean(tdata_2[:, 1]), np.mean(tdata_2[:, 2]), np.mean(tdata_2[:, 3]),
                              np.mean(tdata_3[:, 1]), np.mean(tdata_3[:, 2]), np.mean(tdata_3[:, 3]),
                              np.std(tdata_2[:, 1]), np.std(tdata_2[:, 2]), np.std(tdata_2[:, 3]),
                              np.std(tdata_3[:, 1]), np.std(tdata_3[:, 2]), np.std(tdata_3[:, 3]),
                              stats.skew(tdata_2[:, 1]), stats.skew(tdata_2[:, 2]), stats.skew(tdata_2[:, 3]),
                              stats.skew(tdata_3[:, 1]), stats.skew(tdata_3[:, 2]), stats.skew(tdata_3[:, 3]),
                              stats.kurtosis(tdata_2[:, 1]), stats.kurtosis(tdata_2[:, 2]),
                              stats.kurtosis(tdata_2[:, 3]),
                              stats.kurtosis(tdata_3[:, 1]), stats.kurtosis(tdata_3[:, 2]),
                              stats.kurtosis(tdata_3[:, 3]),
                              np.nansum(tdata_2[:, 1]) - np.nansum(tdata_3[:, 1]),
                              np.nansum(tdata_2[:, 2]) - np.nansum(tdata_3[:, 2]),
                              np.nansum(tdata_2[:, 3]) - np.nansum(tdata_3[:, 3])
                              ])
        except:
            tempx = np.array([dange, np.nansum(tdata_2[:, 1]), np.nansum(tdata_2[:, 2]), np.nansum(tdata_2[:, 3]),
                              np.nansum(tdata_3[:, 1]), np.nansum(tdata_3[:, 2]), np.nansum(tdata_3[:, 3]),
                              np.mean(tdata_2[:, 1]), np.mean(tdata_2[:, 2]), np.mean(tdata_2[:, 3]),
                              np.mean(tdata_3[:, 1]), np.mean(tdata_3[:, 2]), np.mean(tdata_3[:, 3]),
                              np.std(tdata_2[:, 1]), np.std(tdata_2[:, 2]), np.std(tdata_2[:, 3]),
                              np.std(tdata_3[:, 1]), np.std(tdata_3[:, 2]), np.std(tdata_3[:, 3]),
                              stats.skew(tdata_2[:, 1]), stats.skew(tdata_2[:, 2]), stats.skew(tdata_2[:, 3]),
                              stats.skew(tdata_3[:, 1]), stats.skew(tdata_3[:, 2]), stats.skew(tdata_3[:, 3]),
                              0, 0,
                              0, 0, 0, 0,
                              np.nansum(tdata_2[:, 1]) - np.nansum(tdata_3[:, 1]),
                              np.nansum(tdata_2[:, 2]) - np.nansum(tdata_3[:, 2]),
                              np.nansum(tdata_2[:, 3]) - np.nansum(tdata_3[:, 3])
                              ])
        if dange == qiye[0]:
            allx = tempx
        else:
            allx = np.vstack((allx, tempx))
    return allx


data1tezheng = tezheng(df1_1, df1_2, df1_3)

data2tezheng = tezheng(df2_1, df2_2, df2_3)

pd.DataFrame(data1tezheng).to_csv('tezheng1.csv', index=False)
pd.DataFrame(data2tezheng).to_csv('tezheng2.csv', index=False)
