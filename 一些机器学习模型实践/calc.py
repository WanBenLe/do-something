import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit
def woow(a):
    for i in range(len(a)):
        a[i] = str(a[i])
    return a


def srmaen(a):
    srme = np.sum(a[-1, :]) / np.sum(a[-1, :] != 0)
    return srme


def srstd(a):
    b = a[-1, :]
    srsd = np.std(b[b != 0])
    return srsd


np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
fa = []
fb = []
for ii in range(7):
    # 文件路径4
    print(ii)
    data0 = pd.read_csv('model_' + str(ii + 1) + '_resultX.csv').values
    d1 = data0[:, 0]
    d1 = woow(d1)

    allx = np.zeros((10, 4))
    for kk in range(4):
        if np.sum(data0[:, 2 + kk]) != 0.0:
            allx[0, kk] = np.sum(data0[:, 2 + kk]) / 3
            allx[1, kk] = np.std(data0[:, 2 + kk]) * 3 ** 0.5
            allx[2, kk] = allx[0, kk] / allx[1, kk]
            allx[3, kk] = ((data0[-1, 10 + kk]) - 1000000) / (243 * 3)
            allx[4, kk] = ((data0[-1, 10 + kk]) - 1000000) / 1000000
            allx[5, kk] = allx[3, kk] * 243
            allx[6, kk] = allx[3, kk] * 250
            allx[7, kk] = allx[4, kk] / (243 * 3)
            allx[8, kk] = np.std(data0[:, 6 + kk]) * 3 ** 0.5
            allx[9, kk] = allx[6, kk] / allx[8, kk]
    x1 = plt.plot(d1, data0[:, 10], d1, data0[:, 11], d1, data0[:, 12], d1, data0[:, 13])
    plt.legend(handles=x1, labels=['Portfolio 1', 'Portfolio 2', 'Portfolio 3', 'Portfolio 4'],
               loc='best')
    plt.savefig('./Cash' + str(ii + 1) + '.jpg')
    plt.clf()
    x2 = plt.plot(d1, data0[:, 6], d1, data0[:, 7], d1, data0[:, 8], d1, data0[:, 9])
    plt.legend(handles=x2, labels=['Portfolio 1', 'Portfolio 2', 'Portfolio 3', 'Portfolio 4'],
               loc='best')
    plt.savefig('./Return' + str(ii + 1) + '.jpg')
    plt.clf()
    pd.DataFrame(allx).to_csv('mexcel' + str(ii + 1) + '.csv', index=False)
    fa.append(srmaen(allx))
    fb.append(srstd(allx))

famax = np.max(fa)
famin = np.min(fa)
fbmax = np.max(fb)
fbmin = np.min(fb)
for ii in range(7):
    fa[ii] = (fa[ii] - famin) / (famax - famin)
    fb[ii] = (fb[ii] - fbmin) / (fbmax - fbmin)

wall = np.sum([fa, fb], axis=0)
ex1 = []
for ii in range(6):
    ex1.append((wall[-1] - wall[ii]) ** 2)
print(ex1)

print('finished!')
