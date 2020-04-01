import pandas as pd
import numpy as np

data0 = pd.read_csv('data0.csv').values
Illiq = np.zeros((len(data0) - 1, 1))
IlliqSpace = np.zeros((len(data0) - 1, 1))
for i in range(len(data0) - 1):
    Illiq[i] = np.abs(data0[i + 1, 0] - data0[i, 0]) / data0[i, 0] / data0[i, 1] * 10 ** 10

print(np.mean(Illiq))
print(np.percentile(Illiq, 50))
IlliqSpace[Illiq > np.percentile(Illiq, 50)] = 1
IlliqSpace = IlliqSpace.astype(int)
SpceMat = np.zeros((4, 1))
# S11,S12,S21,S22
for i in range(len(IlliqSpace) - 1):
    if IlliqSpace[i] == 0:
        if IlliqSpace[i + 1] == 0:
            SpceMat[0] += 1
        else:
            SpceMat[2] += 1
    else:
        if IlliqSpace[i + 1] == 0:
            SpceMat[1] += 1
# 保证Sum(P)为1
SpceMat[3] = len(IlliqSpace) - np.sum(SpceMat[0:3])
SpceMat = SpceMat / len(IlliqSpace)
print(SpceMat)
print(np.sum(SpceMat))
