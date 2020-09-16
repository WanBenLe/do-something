import pandas as pd
import numpy as np
import matplotlib.pyplot as p

p.rcParams['font.sans-serif'] = ['SimHei']
p.rcParams['axes.unicode_minus'] = False

df1_1 = pd.read_csv('企业信息.csv',encoding='ansi')

print(df1_1.head(10))
dfdeal = df1_1.copy()[['企业代号', '信誉评级', '是否违约']].values
dfdeal[(dfdeal[:, -1] == '是'), -1] = 1
dfdeal[(dfdeal[:, -1] == '否'), -1] = 0
comlen = len(dfdeal)

defalutA = np.sum((dfdeal[:, -2] == 'A') & (dfdeal[:, -1] == 1)) / np.sum(dfdeal[:, -2] == 'A')
defalutB = np.sum((dfdeal[:, -2] == 'B') & (dfdeal[:, -1] == 1)) / np.sum(dfdeal[:, -2] == 'B')
defalutC = np.sum((dfdeal[:, -2] == 'C') & (dfdeal[:, -1] == 1)) / np.sum(dfdeal[:, -2] == 'C')
defalutD = np.sum((dfdeal[:, -2] == 'D') & (dfdeal[:, -1] == 1)) / np.sum(dfdeal[:, -2] == 'D')
print('评级为A的违约率', defalutA)
print('评级为B的违约率', defalutB)
print('评级为C的违约率', defalutC)
print('评级为D的违约率', defalutD)

dfdeal = np.hstack((dfdeal, np.zeros((len(dfdeal), 1))))
dfdeal[(dfdeal[:, -3] == 'A'), -1] = defalutA
dfdeal[(dfdeal[:, -3] == 'B'), -1] = defalutB
dfdeal[(dfdeal[:, -3] == 'C'), -1] = defalutC
dfdeal[(dfdeal[:, -3] == 'D'), -1] = defalutD

pd.DataFrame(dfdeal, columns=['企业代号', '信誉评级', '是否违约', '违约率']).to_csv('truedefault.csv', index=False)


print(1)
