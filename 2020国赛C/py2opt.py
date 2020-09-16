from problem import Problem
from evolution import Evolution
import matplotlib.pyplot as p
import math
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import optimize

p.rcParams['font.sans-serif'] = ['SimHei']
p.rcParams['axes.unicode_minus'] = False

data = pd.read_csv('truedefaultq2.csv').values

df3_1 = pd.read_csv('流失率.csv', encoding='ansi')

print(df3_1.head(10))
df3_rate = df3_1.values
x = df3_rate[:, 0]
yA = df3_rate[:, 1]
yB = df3_rate[:, 2]
yC = df3_rate[:, 3]

p.xlabel(u'利率')
p.ylabel(u'客户流失率')
p.plot(x, yA, 'ro', label=u"原始数据")
p.plot(x, yA, color='b', linestyle='-', label=u"折线图")
tA = interpolate.splrep(x, yA)
xnew = np.linspace(x.min(), x.max(), 300)  # 30
ye = interpolate.splev(xnew, tA)
p.plot(xnew, ye, 'g--', label=u'A三次样条插值')
p.legend()
p.savefig('A评级样条.png')
p.show()
p.clf()

p.xlabel(u'利率')
p.ylabel(u'客户流失率')
p.plot(x, yB, 'ro', label=u"原始数据")
p.plot(x, yB, color='b', linestyle='-', label=u"折线图")
tB = interpolate.splrep(x, yB)
xnew = np.linspace(x.min(), x.max(), 300)  # 30
ye = interpolate.splev(xnew, tB)
p.plot(xnew, ye, 'g--', label=u'B三次样条插值')
p.legend()
p.savefig('B评级样条.png')
p.show()
p.clf()

p.xlabel(u'利率')
p.ylabel(u'客户流失率')
p.plot(x, yC, 'ro', label=u"原始数据")
p.plot(x, yC, color='b', linestyle='-', label=u"折线图")
tC = interpolate.splrep(x, yC)
xnew = np.linspace(x.min(), x.max(), 300)  # 30
ye = interpolate.splev(xnew, tC)
p.plot(xnew, ye, 'g--', label=u'C三次样条插值')
p.legend()
p.savefig('C评级样条.png')
p.show()

kao = 1


def funA(x):
    global tA, kao
    return interpolate.splev(x, tA) - kao


def funB(x):
    global tB, kao
    return interpolate.splev(x, tB) - kao


def funC(x):
    global tC, kao
    return interpolate.splev(x, tC) - kao


def f1(x):
    global data, tA, tB, tC, kao
    lenx = len(data)
    # x[i+lenx]流失率
    # x[i]贷款金额
    lirun = 0.0
    for i in range(lenx):
        sol = 1
        kao = x[i + lenx]
        if data[i, 1] == 'A':
            sol = optimize.root(funA, [0]).x
        elif data[i, 1] == 'B':
            sol = optimize.root(funB, [0]).x
        elif data[i, 1] == 'C':
            sol = optimize.root(funC, [0]).x
        if sol > 0.15:
            sol = 0.15
            continue
        lirun += x[i] * (sol - data[i, -1]) * (1 - x[i + lenx])
    s = -lirun

    return s


def f2(x):
    global data, tA, tB, tC, kao
    lenx = len(data)
    # x[i+lenx]流失率
    # x[i]贷款金额
    lirun = []
    for i in range(lenx):
        kao = x[i + lenx]
        if data[i, 1] == 'A':
            sol = optimize.root(funA, [0]).x
        elif data[i, 1] == 'B':
            sol = optimize.root(funB, [0]).x
        elif data[i, 1] == 'C':
            sol = optimize.root(funC, [0]).x
        else:
            sol = 0.16
            continue

        lirun.append(x[i] * (sol - data[i, -1]) * (1 - x[i + lenx]))
    s = np.var(lirun)
    return s


def f3(x):
    global data, tA, tB, tC, kao
    lenx = len(data)
    # x[i+lenx]流失率
    # x[i]贷款金额
    lirun = []
    for i in range(lenx):
        kao = x[i + lenx]
        if data[i, 1] == 'A':
            sol = optimize.root(funA, [0]).x
        elif data[i, 1] == 'B':
            sol = optimize.root(funB, [0]).x
        elif data[i, 1] == 'C':
            sol = optimize.root(funC, [0]).x
        else:
            sol = 0.16
            continue
        lirun.append(x[i] * (sol - data[i, -1]) * (1 - x[i + lenx]))
    s = -np.mean(lirun) / np.std(lirun)
    return s


var_num = len(data) * 2

rangex = []
for i in range(len(data)):
    rangex.append((10, 100))
for i in range(len(data)):
    rangex.append((0, 0.9))

problem = Problem(num_of_variables=var_num, objectives=[f1, f2, f3], variables_range=rangex,
                  same_range=False,
                  expand=False)
evo = Evolution(problem, num_of_generations=50, mutation_param=30)
func = [i.objectives for i in evo.evolve()]

function1 = [i[0] for i in func]
function2 = [i[1] for i in func]
function3 = [i[2] for i in func]

print('最大总利润:', -function1[-1])
print('最小方差:', function2[-1])
print('最大利润稳健率:', -function3[-1])
print('最终结果:', func[-1])

fig = p.figure()
ax1 = p.axes(projection='3d')  # 设置为3D图
x = np.array(function1) * -1
y = np.array(function2) * -1
z = np.array(function3) * -1
ax1.scatter3D(x, y, z, cmap='Blues')  # 绘制散点图
p.show()
result1 = np.array(evo.population.fronts[0][-1].features[0:len(data)])
result2 = np.array(evo.population.fronts[0][-1].features[len(data):])

resultx = np.hstack((result1.reshape(-1, 1), result2.reshape(-1, 1)))
data = np.hstack((data, resultx))
# ['企业代号', '信誉评级', '是否违约', '违约率','贷款金额','流失率','利率']
data = np.hstack((data, resultx))
data = data[:,0:6]
# 根据流失率得到利率
for i in range(len(data)):
    data[i, -1] = 1
    kao = data[i, -2]
    if data[i, 1] == 'A':
        data[i, -1] = optimize.root(funA, [0]).x[0]
    elif data[i, 1] == 'B':
        data[i, -1] = optimize.root(funB, [0]).x[0]
    elif data[i, 1] == 'C':
        data[i, -1] = optimize.root(funC, [0]).x[0]
    if data[i, -1] > 0.16:
        data[i, -1] = 0.16

# 去到不为D评级的数据
data = data[data[:, 1] != 'D', :]
import joblib
joblib.dump(data,'data.temp')
# 最大额度万为单位
max_edu = 10000
# 将额度调整为最大额度
data[:, -3] = data[:, -3] * max_edu / np.sum(data[:, -3])
# 少于10就不放贷了
data[(data[:, -3] < 10), -3] = 0
true_lirun = []
for i in range(len(data)):
    true_lirun.append(data[i, -3] * (data[i, -1] - data[i, -4]) * (1 - data[i, -2]))
print('结果利润核算：', np.sum(true_lirun))
print('结果利润方差核算：', np.var(true_lirun))
print('结果利润稳健率核算：', np.mean(true_lirun) / np.std(true_lirun))

pd.DataFrame(data, columns=['企业代号', '信誉评级', '违约率', '贷款金额', '流失率', '利率']).to_csv('Q2result.csv', index=False,encoding='ansi')

# plt.xlabel('Function 1', fontsize=15)
# plt.ylabel('Function 2', fontsize=15)
# plt.scatter(function1, function2)
# plt.show()
