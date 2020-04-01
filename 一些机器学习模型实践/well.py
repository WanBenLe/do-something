import pandas as pd
import numpy as np
from numba import jit
import gc

pd.set_option('precision', 9)

# 分组写1
group_bool = 1
# 文件路径
data0 = pd.read_csv('1.csv')
data0['win'] = 0
data0 = data0.values


# 日期数字化
@jit
def date(a):
    for i in range(len(a)):
        t = a[i, 4]
        v = int(t[0:4] + t[5:7] + t[8:10])
        a[i, 4] = v
    return a


data0 = date(data0)
data0_s0 = data0[((data0[:, 3] == '债券买入') | (data0[:, 3] == '费用扣除')), :]


# 筛选
@jit
def s1(a):
    b = np.array([a[0]])
    lenb = 1
    new = 0
    for i in range(len(a) - 1):
        t1 = i + 1
        if a[t1, 3] == '费用扣除':
            b[lenb - 1, 31] = a[t1, 8]
            new = 1
        elif new == 1:
            new = 0
            lenb += 1
            b = np.vstack((b, a[t1, :]))
    return b


data0_s1 = s1(data0_s0)
data0_s1[:, 3] = 0.0


# 计算收益率
@jit
def s2(a):
    for i in range(len(a)):
        a[i, 3] = (a[i, 31] - a[i, 7]) / a[i, 7]
    return a


data0_s2 = s2(data0_s1)


# 获得分组
@jit
def s3(a, b):
    for i in range(len(a)):
        index = np.argwhere(a[i, 1] == b[:, 0])[0][0]
        a[i, 0] = b[index, 1]
    return a


if group_bool == 1:
    data1 = pd.read_csv('A2.csv').values
    data1 = data1[:, [0, 8]]
    data0_s2[:, 0] = 0
    data0_s3 = s3(data0_s2, data1)
else:
    # 没分组的单支视为第一组
    data0_s2[:, 0] = 1
    data0_s3 = data0_s2

data0_s3 = data0_s3[:, [0, 1, 3, 4, 7, 31]]
data0_s3[:, 2] = data0_s3[:, 2] * 100
d4 = pd.DataFrame(data0_s3, columns=['group', 'id', 'return', 'date', 'bf', 'af'])
d4.to_csv('r1.csv', index=False)

data2 = pd.read_csv('dstd.csv').values
data3 = pd.read_csv('shibor.csv', dtype={'date': int, 'night': np.float64}).values


# 获得shibor
@jit
def fun0(a, b):
    for i in range(len(b)):
        index = np.argwhere(b[i, 0] == a[:, 0])[0][0]
        a[index, 1] = b[i, 1]
    return a


dstd = fun0(data2, data3)
dstd[:, 1] = dstd[:, 1] / 100
del data0, data0_s0, data0_s1, data0_s2, data1, data2, data3, date, d4
gc.collect()


# 填充shibor,首先用上一期不为0的填充,最后0用均值填充
@jit
def fun1(a):
    index = np.argwhere(a[:, 1] != 0.0)[0][0]
    for i in range(len(a)):
        if a[i, 1] == 0.0:
            a[i, 1] = a[index, 1]
        else:
            index = i
    a[:, 1][a[:, 1] == 0.0] = np.mean(a[:, 1])
    return a


dstd = fun1(dstd)


@jit
def fun2(a, b):
    group = list(set(b[:, 0]))
    for i in group:
        tb = b[b[:, 0] == i, :]
        x = i + 1
        y = i + 5
        z = i + 9
        for j in range(len(tb)):
            index = np.argwhere(tb[j, 3] == a[:, 0])[0][0]
            a[index, x] = tb[j, 2]
        # shibor转为复利,用大陆法开360次方
        # 接下来用shibor填充为0的部分a
        a[:, x][a[:, x] == 0.0] = a[:, 1][a[:, x] == 0.0]
        # 累计收益率
        a[0, y] = a[0, x]
        # 收益率用百分比防止精度问题的出现
        for j in range(len(a) - 1):
            a[j + 1, y] = (a[j, y] + 100) * (a[j + 1, x] + 100) / 100 - 100
        # 净值
        for j in range(len(a)):
            a[j, z] = 10000 * (a[j, y] + 100)
    return a


allx = fun2(dstd, data0_s3)

cname = ['date0', 'shibor', 'r1', 'r2', 'r3', 'r4', 'cr1', 'cr2', 'cr3', 'cr4', 'c1', 'c2', 'c3', 'c4']
allx = pd.DataFrame(allx, columns=cname)
allx.to_csv('result.csv', index=False)

print('finished!')
